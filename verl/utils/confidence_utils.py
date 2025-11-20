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
Utilities for extracting and computing confidence scores from model outputs.
"""

import torch
import torch.nn.functional as F


def find_confidence_token_positions(
    input_ids: torch.Tensor,  # (bsz, seq_len)
    response_mask: torch.Tensor,  # (bsz, seq_len)
    high_token_id: int,
    low_token_id: int,
):
    """
    在response中定位high或low token的位置，返回它前一个token的位置（用于提取logits）。

    逻辑：
    - 如果response包含 [..., <confidence>, high, </confidence>, ...]
    - high的位置是k，我们需要位置k-1的logits来预测high/low

    Args:
        input_ids: Token IDs, shape (bsz, seq_len)
        response_mask: Mask indicating response tokens, shape (bsz, seq_len)
        high_token_id: Token ID for "high"
        low_token_id: Token ID for "low"

    Returns:
        positions: (bsz,) - 提取logits的位置索引
        valid_mask: (bsz,) - 是否找到confidence token
        is_high: (bsz,) - 是否是high（用于数据验证）
    """
    batch_size, seq_len = input_ids.shape

    # 在response区域内搜索high和low token
    high_mask = (input_ids == high_token_id) & response_mask.bool()  # (bsz, seq_len)
    low_mask = (input_ids == low_token_id) & response_mask.bool()

    # 找到第一个出现的位置
    has_high = high_mask.any(dim=1)  # (bsz,)
    has_low = low_mask.any(dim=1)

    high_positions = high_mask.float().argmax(dim=1)  # (bsz,)
    low_positions = low_mask.float().argmax(dim=1)

    # 优先使用high的位置，如果没有high则用low的位置
    # 注意：input_ids[i, k] = high/low，我们需要 logits[i, k-1]
    conf_token_positions = torch.where(has_high, high_positions, low_positions)
    logits_positions = (conf_token_positions - 1).clamp(min=0)  # 防止负数

    valid_mask = has_high | has_low

    return logits_positions, valid_mask, has_high


def extract_confidence_logits_efficient(
    logits: torch.Tensor,  # (bsz, seq_len, vocab_size)
    positions: torch.Tensor,  # (bsz,) - 要提取的位置
    high_token_id: int,
    low_token_id: int,
    valid_mask: torch.Tensor = None,  # (bsz,) - 可选的有效mask
):
    """
    高效提取特定位置的high/low token logits，不保存完整logits。

    Args:
        logits: Full logits tensor, shape (bsz, seq_len, vocab_size)
        positions: Positions to extract, shape (bsz,)
        high_token_id: Token ID for "high"
        low_token_id: Token ID for "low"
        valid_mask: Optional mask for valid samples, shape (bsz,)

    Returns:
        high_logits: (bsz,) - Logits for high token
        low_logits: (bsz,) - Logits for low token
    """
    batch_size = logits.size(0)
    batch_indices = torch.arange(batch_size, device=logits.device)

    # 提取特定位置的logits: (bsz, vocab_size)
    position_logits = logits[batch_indices, positions, :]

    # 只提取high和low两个token的logits: (bsz,)
    high_logits = position_logits[:, high_token_id]
    low_logits = position_logits[:, low_token_id]

    # 如果提供了valid_mask，将无效样本的logits设为0（避免影响后续计算）
    if valid_mask is not None:
        high_logits = high_logits * valid_mask.float()
        low_logits = low_logits * valid_mask.float()

    return high_logits, low_logits


def compute_confidence_logprobs(
    high_logits: torch.Tensor,  # (bsz,)
    low_logits: torch.Tensor,  # (bsz,)
):
    """
    基于high和low的logits计算log-probabilities。

    使用log-sum-exp技巧来稳定数值计算：
    log P(high) = log(exp(logit_high) / (exp(logit_high) + exp(logit_low)))
                = logit_high - log(exp(logit_high) + exp(logit_low))
                = logit_high - logsumexp([logit_high, logit_low])

    Args:
        high_logits: Logits for high token, shape (bsz,)
        low_logits: Logits for low token, shape (bsz,)

    Returns:
        high_logprobs: Log P(high | context), shape (bsz,)
        low_logprobs: Log P(low | context), shape (bsz,)
    """
    # Stack to (bsz, 2)
    stacked_logits = torch.stack([high_logits, low_logits], dim=-1)

    # Compute logsumexp: (bsz,)
    log_sum_exp = torch.logsumexp(stacked_logits, dim=-1)

    # Compute log probabilities
    high_logprobs = high_logits - log_sum_exp
    low_logprobs = low_logits - log_sum_exp

    return high_logprobs, low_logprobs


def compute_confidence_score(
    high_logprobs: torch.Tensor,  # (bsz,)
    low_logprobs: torch.Tensor,  # (bsz,)
    method: str = "softmax",
):
    """
    基于logprobs计算confidence分数，范围[0, 1]。

    Args:
        high_logprobs: Log P(high), shape (bsz,)
        low_logprobs: Log P(low), shape (bsz,)
        method: "softmax" or "prob_diff"

    Returns:
        scores: Confidence scores in [0, 1], shape (bsz,)
                接近1表示模型更倾向于预测high，接近0表示更倾向于low
    """
    if method == "softmax":
        # sigmoid(log P(high) - log P(low)) = P(high) / (P(high) + P(low))
        logit_diff = high_logprobs - low_logprobs
        scores = torch.sigmoid(logit_diff)
    elif method == "prob_diff":
        # (P(high) - P(low) + 1) / 2，rescale到[0, 1]
        p_high = torch.exp(high_logprobs)
        p_low = torch.exp(low_logprobs)
        scores = (p_high - p_low + 1.0) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'softmax' or 'prob_diff'.")

    return scores


def compute_confidence_loss(
    confidence_scores: torch.Tensor,  # (bsz,)
    target: torch.Tensor,  # (bsz,) - 可以是[0,1]连续值或{0,1}二分类标签
    loss_type: str = "regression",
    valid_mask: torch.Tensor = None,  # (bsz,) - 有效样本mask
):
    """
    计算confidence预测损失。

    Args:
        confidence_scores: Predicted confidence scores in [0, 1], shape (bsz,)
        target: Ground truth confidence values, shape (bsz,)
                - regression: [0, 1]连续值
                - classification: {0, 1}二分类标签
        loss_type: "regression" or "classification"
        valid_mask: Optional mask for valid samples, shape (bsz,)

    Returns:
        loss: Scalar loss value
    """
    if valid_mask is not None:
        # 只计算有效样本的损失
        confidence_scores = confidence_scores[valid_mask]
        target = target[valid_mask]

        if confidence_scores.numel() == 0:
            # 没有有效样本，返回0损失
            return torch.tensor(0.0, device=confidence_scores.device)

    if loss_type == "regression":
        # MSE loss
        loss = F.mse_loss(confidence_scores, target)
    elif loss_type == "classification":
        # Binary cross-entropy loss
        target_float = target.float()
        loss = F.binary_cross_entropy(confidence_scores, target_float)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'regression' or 'classification'.")

    return loss


def get_confidence_token_ids(tokenizer, high_text: str = "high", low_text: str = "low"):
    """
    从tokenizer获取high和low的token IDs。

    注意：需要确保tokenizer将"high"和"low"编码为单个token。
    如果不是，需要调整提取逻辑或修改prompt格式。

    Args:
        tokenizer: HuggingFace tokenizer
        high_text: Text for high confidence
        low_text: Text for low confidence

    Returns:
        high_token_id: int
        low_token_id: int
    """
    high_tokens = tokenizer.encode(high_text, add_special_tokens=False)
    low_tokens = tokenizer.encode(low_text, add_special_tokens=False)

    if len(high_tokens) != 1:
        raise ValueError(
            f"Expected '{high_text}' to be encoded as a single token, "
            f"but got {len(high_tokens)} tokens: {high_tokens}. "
            f"Consider adjusting the prompt format or tokenizer vocabulary."
        )

    if len(low_tokens) != 1:
        raise ValueError(
            f"Expected '{low_text}' to be encoded as a single token, "
            f"but got {len(low_tokens)} tokens: {low_tokens}. "
            f"Consider adjusting the prompt format or tokenizer vocabulary."
        )

    return high_tokens[0], low_tokens[0]
