# Confidence-Aware DAPO Training

This extension adds confidence loss to DAPO training for models that output confidence tags like `<confidence>high</confidence>` or `<confidence>low</confidence>`.

## Overview

The confidence loss helps the model learn to accurately predict its confidence level by:
1. Extracting log-probabilities for "high" and "low" tokens from the model output
2. Computing a confidence score in [0, 1] range
3. Comparing the predicted confidence with ground truth (regression or classification)
4. Adding this auxiliary loss to the main DAPO policy gradient loss

## Quick Start

### Step 1: Get Token IDs

First, find the token IDs for "high" and "low" in your tokenizer:

```bash
python recipe/dapo/get_confidence_token_ids.py --model_path /path/to/your/model
```

This will output something like:
```
✓ 'high' is a single token: 5487
✓ 'low' is a single token: 3347

Configuration for dapo_confidence_trainer.yaml:
==========================================
actor_rollout_ref:
  actor:
    use_confidence_loss: true
    conf_loss_type: regression
    conf_loss_coef: 0.1
    conf_score_method: softmax
    high_token_id: 5487
    low_token_id: 3347
```

### Step 2: Prepare Your Data

Your training data should include confidence labels. Two formats are supported:

**Regression format** (continuous confidence values):
```python
{
    "prompt": "What is 2+2?",
    "response": "<confidence>high</confidence> The answer is 4.",
    "target_confidence": 0.95  # Value in [0, 1]
}
```

**Classification format** (binary labels):
```python
{
    "prompt": "What is the meaning of life?",
    "response": "<confidence>low</confidence> It's hard to say definitively.",
    "confidence_label": 0  # 0 = low, 1 = high
}
```

**Important**: All responses must contain exactly one `<confidence>high</confidence>` or `<confidence>low</confidence>` tag.

### Step 3: Configure Training

Copy the example config and fill in your token IDs:

```bash
cp recipe/dapo/config/dapo_confidence_trainer.yaml recipe/dapo/config/my_config.yaml
```

Edit `my_config.yaml`:

```yaml
actor_rollout_ref:
  actor:
    use_confidence_loss: true
    conf_loss_type: regression  # or "classification"
    conf_loss_coef: 0.1  # Weight of confidence loss
    conf_score_method: softmax  # or "prob_diff"
    high_token_id: 5487  # Replace with your token ID
    low_token_id: 3347   # Replace with your token ID
```

### Step 4: Run Training

```bash
python -m verl.trainer.main_ppo \
    --config-name my_config \
    data.train_files=/path/to/your/data.parquet \
    actor_rollout_ref.model.path=/path/to/your/model \
    [other DAPO configs...]
```

## Configuration Options

### `use_confidence_loss` (bool, default: False)
Enable confidence loss. Set to `true` to activate.

### `conf_loss_type` (str, default: "regression")
Type of confidence loss:
- `"regression"`: MSE loss for continuous confidence values in [0, 1]
- `"classification"`: Binary cross-entropy loss for {0, 1} labels

### `conf_loss_coef` (float, default: 0.1)
Weight coefficient for confidence loss relative to policy gradient loss.
- Start with small values (0.01-0.1)
- Increase if confidence predictions are not improving
- Decrease if main task performance degrades

### `conf_score_method` (str, default: "softmax")
Method for computing confidence score from log-probabilities:
- `"softmax"`: `sigmoid(log P(high) - log P(low))` - Recommended
- `"prob_diff"`: `(P(high) - P(low) + 1) / 2` - Alternative

### `high_token_id` (int, required)
Token ID for the word "high" in your tokenizer.

### `low_token_id` (int, required)
Token ID for the word "low" in your tokenizer.

## Monitoring Training

The following metrics are logged during training:

- `actor/conf_loss`: Confidence loss value
- `actor/conf_score_mean`: Average predicted confidence score
- `actor/conf_valid_ratio`: Fraction of samples with valid confidence tags
- `actor/pg_loss`: Main policy gradient loss (should remain stable)

Example logs:
```
Step 100: actor/conf_loss=0.023, actor/conf_score_mean=0.68, actor/conf_valid_ratio=0.98
```

## Troubleshooting

### Issue: "high" or "low" is not a single token

**Solution**: Use the script to check tokenization:
```bash
python recipe/dapo/get_confidence_token_ids.py --model_path /path/to/model
```

If they're multi-token, consider:
1. Using different confidence indicators (e.g., "H"/"L", "yes"/"no")
2. Adding these words to the tokenizer vocabulary
3. Modifying the prompt format

### Issue: `conf_valid_ratio` is low

**Cause**: Many samples don't have valid confidence tags.

**Solution**:
1. Check your data format - ensure all responses have `<confidence>` tags
2. Verify the tags are properly formatted
3. Check tokenization with the helper script

### Issue: Main task performance degrades

**Cause**: Confidence loss coefficient is too high.

**Solution**: Reduce `conf_loss_coef` from 0.1 to 0.01 or 0.001.

### Issue: Confidence predictions not improving

**Cause**: Confidence loss coefficient is too low.

**Solution**: Increase `conf_loss_coef` to 0.2-0.5.

### Issue: "Failed to extract confidence logits" warning

**Cause**: The code path doesn't support confidence extraction (e.g., using remove_padding or fused_kernels).

**Solution**: Currently, confidence extraction works best with:
- `use_remove_padding: false` (or standard tensor format)
- `use_fused_kernels: false`

Support for other configurations will be added in future updates.

## Implementation Details

### How It Works

1. **Forward Pass**: During model forward, we identify the position of "high" or "low" tokens in each response
2. **Logits Extraction**: Extract only the logits for high_token_id and low_token_id at those positions (memory efficient!)
3. **Score Computation**: Compute `P(high)` and `P(low)`, then derive confidence score
4. **Loss Calculation**: Compare with ground truth and compute MSE or BCE loss
5. **Backpropagation**: Gradient flows through logits → model parameters

### Memory Efficiency

Unlike storing full logits tensors `(batch_size, seq_len, vocab_size)`, we only extract and store:
- `high_logits`: `(batch_size,)`
- `low_logits`: `(batch_size,)`

This reduces memory overhead to ~0.01% of the full logits!

### Compatibility

- ✅ Works with DAPO's dynamic sampling and group filtering
- ✅ Works with DAPO's separated clipping
- ✅ Compatible with standard PPO, GRPO, and other RL algorithms
- ⚠️ Currently requires standard tensor format (not nested/rmpad in some cases)

## Example: Full Training Command

```bash
python -m verl.trainer.main_ppo \
    --config-name dapo_confidence_trainer \
    data.train_files=~/data/my_dataset.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.use_confidence_loss=true \
    actor_rollout_ref.actor.conf_loss_type=regression \
    actor_rollout_ref.actor.conf_loss_coef=0.1 \
    actor_rollout_ref.actor.high_token_id=5487 \
    actor_rollout_ref.actor.low_token_id=3347 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=acc \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3
```

## Citation

If you use this feature, please cite both VERL and DAPO:

```bibtex
@article{sheng2024hybridflow,
  title={HybridFlow: A Flexible and Efficient RLHF Framework},
  author={Sheng, Guangming and others},
  journal={arXiv preprint arXiv:2409.19256},
  year={2024}
}
```

## Support

For issues or questions:
1. Check this README
2. Run the helper script to verify token IDs
3. Check training logs for warning messages
4. Open an issue on GitHub: https://github.com/volcengine/verl/issues
