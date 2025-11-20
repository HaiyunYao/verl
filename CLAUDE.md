# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training framework for Large Language Models. It implements the HybridFlow architecture, which uses a hybrid-controller programming model to enable flexible representation and efficient execution of complex post-training dataflows.

Key features:
- Supports multiple training backends: **FSDP**, **FSDP2**, **Megatron-LM**
- Supports multiple inference backends: **vLLM**, **SGLang**, **HF Transformers**
- Implements various RL algorithms: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, etc.
- Uses **Ray** for distributed training orchestration
- Uses **Hydra** for hierarchical configuration management
- Scales to 671B parameter models with expert parallelism

## Installation & Development Setup

**For development:**
```bash
# With vLLM backend
pip install -e .[test,vllm]

# With SGLang backend
pip install -e .[test,sglang]

# Install pre-commit hooks for code formatting
pip install pre-commit
pre-commit install
```

**Available extras:** `test`, `prime`, `geo`, `gpu`, `math`, `vllm`, `sglang`, `trl`, `mcore`, `transferqueue`

## Running Tests

Tests are organized into different categories:

```bash
# Run unit tests with pytest
pytest tests/

# Run end-to-end tests (require GPUs)
bash tests/special_e2e/<test_script>.sh

# Run specific test files
pytest tests/interactions/test_gsm8k_interaction.py
```

Test workflows are defined in `.github/workflows/` for:
- GPU unit tests
- CPU unit tests
- vLLM integration tests
- SGLang integration tests

## Code Formatting & Linting

Uses `ruff` via pre-commit hooks:

```bash
# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files ruff
```

Configuration in `pyproject.toml`:
- Line length: 120 characters
- Imports sorted with isort (first-party: `verl`)

## Running Training

All training scripts use Hydra configuration. Main entry points:

### PPO Training
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=<path> \
    actor_rollout_ref.model.path=<model_path> \
    actor_rollout_ref.rollout.name=vllm \
    trainer.n_gpus_per_node=8 \
    [additional overrides...]
```

### GRPO Training
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    [other configs...]
```

### SFT (Supervised Fine-Tuning)
```bash
python3 -m verl.trainer.fsdp_sft_trainer \
    [configs...]
```

**Example scripts:** Check `examples/ppo_trainer/`, `examples/grpo_trainer/`, `examples/sft/` for reference implementations.

## Architecture Overview

### Core Components

**1. Trainer Module (`verl/trainer/`)**
- `main_ppo.py`: Main PPO training loop
- `main_generation.py`: Generation/inference script
- `fsdp_sft_trainer.py`: Supervised fine-tuning
- `ppo/`: PPO-specific algorithms and logic
- `config/`: Hydra configuration schemas (YAML files)

**2. Workers Module (`verl/workers/`)**

Implements distributed Ray workers for different roles:
- `actor/`: Policy model training workers (FSDP/Megatron)
- `rollout/`: Generation workers (vLLM/SGLang/HF)
- `critic/`: Value model training workers
- `reward_model/`: Reward model inference workers
- `fsdp_workers.py`: FSDP backend implementations
- `megatron_workers.py`: Megatron-LM backend implementations

**3. Models Module (`verl/models/`)**

Model implementations and integrations:
- FSDP model wrappers
- Megatron model wrappers
- Reward model implementations

**4. Utils Module (`verl/utils/`)**

Common utilities for distributed training, data processing, etc.

**5. Protocol (`verl/protocol.py`)**

Defines data structures and communication protocols between workers.

### Training Flow

The hybrid-controller architecture separates training into distinct phases:

1. **Rollout Phase**: Generate responses using inference engines (vLLM/SGLang)
   - Model loaded in inference-optimized format
   - Parallel generation across prompts

2. **Training Phase**: Update policy and value models
   - Model resharded for training (FSDP/Megatron)
   - Compute advantages and policy gradients
   - Update actor (policy) and critic (value) networks

3. **Reward Computation**: Calculate rewards for responses
   - Function-based rewards (e.g., code execution, math verification)
   - Model-based rewards (reward model inference)

The **3D-HybridEngine** efficiently manages model resharding between these phases.

### Configuration System

Uses Hydra with hierarchical configs in `verl/trainer/config/`:

- `ppo_trainer.yaml`: Main PPO configuration
- `ppo_megatron_trainer.yaml`: PPO with Megatron backend
- `sft_trainer.yaml`: SFT configuration
- `actor/`, `critic/`, `rollout/`, `reward_model/`: Component configs
- `algorithm/`, `data/`, `optim/`: Algorithm-specific configs

Override configs from command line:
```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=/path/to/model \
    data.train_batch_size=1024
```

### Backend Integration

**FSDP/FSDP2 (Training):**
- Implemented in `verl/workers/fsdp_workers.py`
- Supports parameter/optimizer offloading
- Enable FSDP2: `actor_rollout_ref.actor.strategy=fsdp2`

**Megatron-LM (Training):**
- Implemented in `verl/workers/megatron_workers.py`
- Supports tensor/pipeline/expert parallelism
- Required for very large models (e.g., DeepSeek 671B)

**vLLM (Inference):**
- Set `actor_rollout_ref.rollout.name=vllm`
- Version: `>=0.8.5,<=0.11.0` (avoid 0.7.x)
- Configured via `actor_rollout_ref.rollout.*` parameters

**SGLang (Inference):**
- Set `actor_rollout_ref.rollout.name=sglang`
- Version: `==0.5.5`
- Supports multi-turn rollout and tool calling

## Data Preparation

Data processing scripts in `examples/data_preprocess/`:
```bash
# GSM8K dataset
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# MATH dataset
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
```

Data format: Parquet files with columns specified per dataset.

## Recipe Directory

The `recipe/` directory contains research implementations:
- `dapo/`: DAPO algorithm
- `prime/`: PRIME algorithm
- `gspo/`: GSPO algorithm
- `entropy/`: Entropy mechanism studies
- Multi-turn and tool-calling examples

These are experimental features and may require additional dependencies.

## Common Commands

```bash
# Install for development
pip install -e .[test,vllm]

# Format code
pre-commit run --all-files

# Run PPO training (8 GPUs, single node)
python3 -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    [overrides...]

# Run tests
pytest tests/

# Build documentation
cd docs && pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/
```

## Important Development Notes

1. **Distributed Training**: All training uses Ray for distribution. Workers are allocated via Ray resource specifications.

2. **Configuration Precedence**: Command-line overrides > Hydra config files > defaults

3. **Model Paths**: Can be HuggingFace model IDs or local paths. Models are auto-downloaded if needed.

4. **GPU Memory**: Tune `gpu_memory_utilization`, `micro_batch_size_per_gpu`, and offloading settings for OOM issues.

5. **Sequence Packing**: Enable with appropriate data config for better GPU utilization.

6. **LoRA Training**: Supported via PEFT integration for memory-efficient training.

7. **Multi-turn Rollout**: See `examples/sglang_multiturn/` for multi-turn conversation training.

8. **Reward Functions**: Implement custom rewards in `verl/trainer/ppo/reward/` or use function-based rewards.

## Performance Tuning

Key parameters for optimization:
- `data.train_batch_size`: Global batch size
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Training micro-batch
- `actor_rollout_ref.rollout.gpu_memory_utilization`: vLLM/SGLang memory
- `actor_rollout_ref.rollout.tensor_model_parallel_size`: Inference TP size
- Enable gradient checkpointing: `enable_gradient_checkpointing=True`
- Enable sequence packing for better throughput

See documentation at https://verl.readthedocs.io/en/latest/perf/perf_tuning.html

## Testing Changes

When adding features:
1. Add unit tests in `tests/`
2. Update relevant CI workflow in `.github/workflows/`
3. Add end-to-end test script in `tests/special_e2e/` if needed
4. Minimize test workload (small models, few steps)

## Resources

- Documentation: https://verl.readthedocs.io
- Paper: https://arxiv.org/abs/2409.19256
- Issues: https://github.com/volcengine/verl/issues
- Slack: verl-project workspace
