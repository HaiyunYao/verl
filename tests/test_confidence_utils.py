#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for confidence loss utilities.

Usage:
    python -m pytest tests/test_confidence_utils.py -v
    # or
    python tests/test_confidence_utils.py
"""

import torch

from verl.utils.confidence_utils import (
    compute_confidence_logprobs,
    compute_confidence_loss,
    compute_confidence_score,
    extract_confidence_logits_efficient,
    find_confidence_token_positions,
)


def test_find_confidence_token_positions():
    """Test finding confidence token positions in input_ids."""
    # Setup
    batch_size = 4
    seq_len = 20
    high_token_id = 100
    low_token_id = 200

    # Create mock input_ids with high/low at different positions
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    input_ids[0, 10] = high_token_id  # Sample 0 has "high" at position 10
    input_ids[1, 15] = low_token_id  # Sample 1 has "low" at position 15
    input_ids[2, 8] = high_token_id  # Sample 2 has "high" at position 8
    input_ids[3, 5] = low_token_id  # Sample 3 has "low" at position 5

    # Create response mask (assume last 10 tokens are response)
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, 5:15] = 1  # Response region

    # Extract positions
    positions, valid_mask, is_high = find_confidence_token_positions(
        input_ids=input_ids,
        response_mask=response_mask,
        high_token_id=high_token_id,
        low_token_id=low_token_id,
    )

    # Verify
    assert positions.shape == (batch_size,)
    assert valid_mask.shape == (batch_size,)
    assert is_high.shape == (batch_size,)

    # Check positions are correct (should be one less than actual token position)
    assert positions[0] == 9  # high at 10, so position is 9
    assert valid_mask[0] == True
    assert is_high[0] == True

    assert positions[1] == 14  # low at 15, so position is 14
    assert valid_mask[1] == True
    assert is_high[1] == False

    print("✓ test_find_confidence_token_positions passed")


def test_extract_confidence_logits():
    """Test extracting confidence logits from full logits tensor."""
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    high_token_id = 100
    low_token_id = 200

    # Create mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Set some specific values for testing
    logits[0, 5, high_token_id] = 10.0
    logits[0, 5, low_token_id] = 2.0
    logits[1, 10, high_token_id] = 3.0
    logits[1, 10, low_token_id] = 8.0

    # Positions to extract
    positions = torch.tensor([5, 10, 7, 3])
    valid_mask = torch.tensor([True, True, True, False])

    # Extract
    high_logits, low_logits = extract_confidence_logits_efficient(
        logits=logits, positions=positions, high_token_id=high_token_id, low_token_id=low_token_id, valid_mask=valid_mask
    )

    # Verify shapes
    assert high_logits.shape == (batch_size,)
    assert low_logits.shape == (batch_size,)

    # Verify values
    assert torch.isclose(high_logits[0], torch.tensor(10.0))
    assert torch.isclose(low_logits[0], torch.tensor(2.0))
    assert torch.isclose(high_logits[1], torch.tensor(3.0))
    assert torch.isclose(low_logits[1], torch.tensor(8.0))

    # Invalid sample should be zero
    assert high_logits[3] == 0.0
    assert low_logits[3] == 0.0

    print("✓ test_extract_confidence_logits passed")


def test_compute_confidence_logprobs():
    """Test computing log-probabilities from logits."""
    batch_size = 3
    high_logits = torch.tensor([10.0, 3.0, 5.0])
    low_logits = torch.tensor([2.0, 8.0, 5.0])

    high_logprobs, low_logprobs = compute_confidence_logprobs(high_logits, low_logits)

    # Verify shapes
    assert high_logprobs.shape == (batch_size,)
    assert low_logprobs.shape == (batch_size,)

    # Verify they sum to 1 in probability space
    for i in range(batch_size):
        p_high = torch.exp(high_logprobs[i])
        p_low = torch.exp(low_logprobs[i])
        total_prob = p_high + p_low
        assert torch.isclose(total_prob, torch.tensor(1.0), atol=1e-5)

    # Verify relative magnitudes
    assert high_logprobs[0] > low_logprobs[0]  # 10 > 2, so P(high) > P(low)
    assert high_logprobs[1] < low_logprobs[1]  # 3 < 8, so P(high) < P(low)
    assert torch.isclose(high_logprobs[2], low_logprobs[2], atol=1e-5)  # Equal logits

    print("✓ test_compute_confidence_logprobs passed")


def test_compute_confidence_score():
    """Test computing confidence scores from log-probabilities."""
    batch_size = 3
    high_logprobs = torch.tensor([-0.1, -2.0, -0.693])  # log probabilities
    low_logprobs = torch.tensor([-2.3, -0.1, -0.693])

    # Test softmax method
    scores_softmax = compute_confidence_score(high_logprobs, low_logprobs, method="softmax")

    assert scores_softmax.shape == (batch_size,)
    assert torch.all((scores_softmax >= 0) & (scores_softmax <= 1))

    # When high_logprob > low_logprob, score should be > 0.5
    assert scores_softmax[0] > 0.5
    assert scores_softmax[1] < 0.5
    assert torch.isclose(scores_softmax[2], torch.tensor(0.5), atol=1e-3)

    # Test prob_diff method
    scores_prob_diff = compute_confidence_score(high_logprobs, low_logprobs, method="prob_diff")

    assert scores_prob_diff.shape == (batch_size,)
    assert torch.all((scores_prob_diff >= 0) & (scores_prob_diff <= 1))

    print("✓ test_compute_confidence_score passed")


def test_compute_confidence_loss_regression():
    """Test computing regression loss for confidence."""
    batch_size = 4
    confidence_scores = torch.tensor([0.9, 0.7, 0.3, 0.5])
    target = torch.tensor([0.95, 0.65, 0.4, 0.6])
    valid_mask = torch.tensor([True, True, True, False])

    # Compute loss
    loss = compute_confidence_loss(
        confidence_scores=confidence_scores, target=target, loss_type="regression", valid_mask=valid_mask
    )

    # Verify
    assert loss.ndim == 0  # Scalar
    assert loss > 0

    # Manual calculation for first 3 samples (last is invalid)
    expected_loss = ((0.9 - 0.95) ** 2 + (0.7 - 0.65) ** 2 + (0.3 - 0.4) ** 2) / 3
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5)

    print("✓ test_compute_confidence_loss_regression passed")


def test_compute_confidence_loss_classification():
    """Test computing classification loss for confidence."""
    batch_size = 4
    confidence_scores = torch.tensor([0.9, 0.7, 0.3, 0.5])
    target = torch.tensor([1, 1, 0, 1])  # Binary labels
    valid_mask = torch.tensor([True, True, True, False])

    # Compute loss
    loss = compute_confidence_loss(
        confidence_scores=confidence_scores, target=target, loss_type="classification", valid_mask=valid_mask
    )

    # Verify
    assert loss.ndim == 0  # Scalar
    assert loss > 0

    print("✓ test_compute_confidence_loss_classification passed")


def test_end_to_end_confidence_extraction():
    """End-to-end test simulating the full confidence extraction pipeline."""
    batch_size = 2
    seq_len = 15
    vocab_size = 500
    high_token_id = 50
    low_token_id = 100

    # Create mock data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids[0, 8] = high_token_id  # Sample 0: high confidence
    input_ids[1, 10] = low_token_id  # Sample 1: low confidence

    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, 5:] = 1

    logits = torch.randn(batch_size, seq_len, vocab_size)
    # Make high more likely for sample 0
    logits[0, 7, high_token_id] = 10.0
    logits[0, 7, low_token_id] = 2.0
    # Make low more likely for sample 1
    logits[1, 9, high_token_id] = 3.0
    logits[1, 9, low_token_id] = 9.0

    # Step 1: Find positions
    positions, valid_mask, is_high = find_confidence_token_positions(
        input_ids, response_mask, high_token_id, low_token_id
    )

    assert valid_mask.all()
    assert is_high[0] == True
    assert is_high[1] == False

    # Step 2: Extract logits
    high_logits, low_logits = extract_confidence_logits_efficient(
        logits, positions, high_token_id, low_token_id, valid_mask
    )

    # Step 3: Compute logprobs
    high_logprobs, low_logprobs = compute_confidence_logprobs(high_logits, low_logits)

    # Step 4: Compute scores
    scores = compute_confidence_score(high_logprobs, low_logprobs, method="softmax")

    assert scores[0] > 0.5  # Sample 0 should be confident (high)
    assert scores[1] < 0.5  # Sample 1 should not be confident (low)

    # Step 5: Compute loss
    target = torch.tensor([0.9, 0.2])  # Ground truth confidence
    loss = compute_confidence_loss(scores, target, loss_type="regression", valid_mask=valid_mask)

    assert loss > 0
    assert loss.requires_grad  # Should be differentiable

    print("✓ test_end_to_end_confidence_extraction passed")


if __name__ == "__main__":
    print("Running confidence utils tests...\n")

    test_find_confidence_token_positions()
    test_extract_confidence_logits()
    test_compute_confidence_logprobs()
    test_compute_confidence_score()
    test_compute_confidence_loss_regression()
    test_compute_confidence_loss_classification()
    test_end_to_end_confidence_extraction()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
