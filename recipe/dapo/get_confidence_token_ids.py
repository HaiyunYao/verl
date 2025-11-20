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
Helper script to get token IDs for confidence words ("high" and "low") from a tokenizer.

Usage:
    python get_confidence_token_ids.py --model_path /path/to/model

This script helps you configure the confidence loss by finding the token IDs
that your tokenizer uses for "high" and "low" words.
"""

import argparse

from transformers import AutoTokenizer


def get_confidence_token_ids(model_path: str, high_text: str = "high", low_text: str = "low"):
    """
    Get token IDs for confidence words from a tokenizer.

    Args:
        model_path: Path to model or HuggingFace model ID
        high_text: Text for high confidence (default: "high")
        low_text: Text for low confidence (default: "low")

    Returns:
        Tuple of (high_token_id, low_token_id)
    """
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Encode high text
    high_tokens = tokenizer.encode(high_text, add_special_tokens=False)
    print(f"\nEncoding '{high_text}':")
    print(f"  Token IDs: {high_tokens}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in high_tokens]}")

    # Encode low text
    low_tokens = tokenizer.encode(low_text, add_special_tokens=False)
    print(f"\nEncoding '{low_text}':")
    print(f"  Token IDs: {low_tokens}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in low_tokens]}")

    # Check if single token
    if len(high_tokens) != 1:
        print(f"\n⚠️  WARNING: '{high_text}' is encoded as {len(high_tokens)} tokens, not 1!")
        print("   This may cause issues with confidence extraction.")
        print("   Consider using a different word or modifying your prompt format.")
    else:
        print(f"\n✓ '{high_text}' is a single token: {high_tokens[0]}")

    if len(low_tokens) != 1:
        print(f"\n⚠️  WARNING: '{low_text}' is encoded as {len(low_tokens)} tokens, not 1!")
        print("   This may cause issues with confidence extraction.")
        print("   Consider using a different word or modifying your prompt format.")
    else:
        print(f"\n✓ '{low_text}' is a single token: {low_tokens[0]}")

    # Test with full tag
    full_high_tag = f"<confidence>{high_text}</confidence>"
    full_low_tag = f"<confidence>{low_text}</confidence>"

    print(f"\n\nTesting full tags:")
    print(f"  '{full_high_tag}':")
    high_tag_tokens = tokenizer.encode(full_high_tag, add_special_tokens=False)
    print(f"    Token IDs: {high_tag_tokens}")
    print(f"    Decoded: {[tokenizer.decode([t]) for t in high_tag_tokens]}")

    print(f"\n  '{full_low_tag}':")
    low_tag_tokens = tokenizer.encode(full_low_tag, add_special_tokens=False)
    print(f"    Token IDs: {low_tag_tokens}")
    print(f"    Decoded: {[tokenizer.decode([t]) for t in low_tag_tokens]}")

    # Provide configuration snippet
    print("\n" + "=" * 80)
    print("Configuration for dapo_confidence_trainer.yaml:")
    print("=" * 80)

    if len(high_tokens) == 1 and len(low_tokens) == 1:
        print(f"""
actor_rollout_ref:
  actor:
    use_confidence_loss: true
    conf_loss_type: regression  # or "classification"
    conf_loss_coef: 0.1
    conf_score_method: softmax
    high_token_id: {high_tokens[0]}
    low_token_id: {low_tokens[0]}
""")
    else:
        print("\n⚠️  Cannot generate configuration due to multi-token words.")
        print("   Please choose different confidence indicators that are single tokens.")

    return high_tokens[0] if len(high_tokens) == 1 else None, low_tokens[0] if len(low_tokens) == 1 else None


def main():
    parser = argparse.ArgumentParser(description="Get token IDs for confidence words")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--high_text",
        type=str,
        default="high",
        help="Text for high confidence (default: high)",
    )
    parser.add_argument(
        "--low_text",
        type=str,
        default="low",
        help="Text for low confidence (default: low)",
    )

    args = parser.parse_args()

    get_confidence_token_ids(args.model_path, args.high_text, args.low_text)


if __name__ == "__main__":
    main()
