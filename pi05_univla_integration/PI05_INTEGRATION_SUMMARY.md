# PI05 Integration Summary

## Overview

This document summarizes the integration of PI05 (PaliGemma-2B) VLM into the UniVLA training pipeline for pretraining on Bridge dataset and fine-tuning on LIBERO tasks.

## Changes Made

### 1. PI05 Model Implementation (`prismatic/models/vlas/pi05_for_univla.py`)

**Key Changes:**
- Added `num_patches` property to `VisionBackboneWrapper` for training compatibility
- Added `all_module_keys` and `trainable_module_keys` attributes for training infrastructure
- Modified `forward()` method to return `CausalLMOutputWithPast` instead of dict
- Added `compute_loss_with_openvla_tokens()` method for loss computation with OpenVLA token mapping
- Made output compatible with UniVLA's `run_latent_action_training()` method

**Architecture:**
- Vision: SigLIP encoder (1152 hidden dim, frozen)
- Language: Gemma-2B (2048 hidden dim, frozen)
- LAM (VQ-VAE): Vocab size 512, 4 latent tokens, frozen
- Trainable: Projector + Sequence Head (~34.6M params)

### 2. Configuration (`prismatic/conf/vla.py`)

**Existing Configuration:**
- `Exp_PI05_224px_Bridge` class added
- Uses `pi05-224px+7b` as `base_vlm`
- Sets proper freezing strategy (vision and LLM frozen, only adapter trainable)

### 3. Fine-tuning Scripts

#### 3a. Main Fine-tuning Script (`baseline_scripts/run_finetune_pi05.sh`)

**New Script Created:**
- Added `--use_pi05` flag (default: false)
- Added PI05-specific parameters:
  - `--pi05_lam_vocab_size` (default: 512)
  - `--pi05_lam_num_tokens` (default: 4)
  - `--pi05_use_multi_token_prediction` (default: true)
- Supports all original parameters plus new PI05 flags

**Usage:**
```bash
# OpenVLA (default):
./baseline_scripts/run_finetune_pi05.sh

# PI05:
./baseline_scripts/run_finetune_pi05.sh --use_pi05 --pi05_lam_vocab_size 512 --pi05_lam_num_tokens 4
```

#### 3b. PI05-Specific Fine-tuning Script (`vla-scripts/finetune_libero_pi05.py`)

**Existing Features:**
- `use_pi05` flag in configuration
- Conditional model loading (PI05 vs OpenVLA)
- Modified forward pass for PI05 with token mapping
- Simplified checkpoint saving for PI05

### 4. Pretraining Script (`vla-scripts/train_pi05_bridge.py`)

**Features:**
- Uses `get_latent_vla_dataset_and_collator()` for Bridge dataset
- Uses `get_train_strategy()` for training loop
- Freezes vision and LLM backbones, trains only adapter
- Properly integrates LAM for action encoding
- Compatible with UniVLA's training infrastructure

**Usage:**
```bash
# Single GPU:
CUDA_VISIBLE_DEVICES=0 python vla-scripts/train_pi05_bridge.py

# Multi-GPU:
torchrun --nproc_per_node=1 vla-scripts/train_pi05_bridge.py
```

### 5. Evaluation

**Note:** The existing evaluation script (`experiments/robot/libero/run_libero_eval.py`) needs to be adapted for PI05. This requires:
- Loading PI05 model checkpoint format
- Handling PI05's multi-token action prediction
- Modified action decoding for PI05

## Dataset Locations

- **Bridge Dataset:** `/data` (for pretraining)
- **LIBERO Dataset:** `/LIBERO/modified_libero_rlds` (for fine-tuning)

## Testing

**Model Loading Test:**
Successfully verified that PI05 model loads correctly with:
- Vision embed dim: 1152
- Num patches: 256
- All module keys: ['vision_encoder', 'language_model', 'projector', 'sequence_head']
- Trainable module keys: ['projector', 'sequence_head']

## Next Steps

1. **Run Pretraining:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python vla-scripts/train_pi05_bridge.py
   ```

2. **Run Fine-tuning on LIBERO:**
   ```bash
   conda activate embodx
   ./baseline_scripts/run_finetune_pi05.sh --use_pi05
   ```

3. **Run Evaluation:**
   ```bash
   conda activate embodx
   ./baseline_scripts/run_eval.sh  # (Need to adapt for PI05 evaluation)
   ```

## File Structure

```
UniVLA/
├── baseline_scripts/
│   ├── run_eval.sh           # Evaluation script (needs PI05 adaptation)
│   ├── run_finetune.sh         # Original fine-tuning (OpenVLA only)
│   └── run_finetune_pi05.sh   # New fine-tuning with PI05 support
├── vla-scripts/
│   ├── finetune_libero.py       # Main fine-tuning (adapted for PI05)
│   ├── finetune_libero_pi05.py  # PI05-specific fine-tuning
│   ├── train_pi05_bridge.py      # Pretraining script for PI05 on Bridge
│   └── train.py                  # Original pretraining (for reference)
├── prismatic/
│   ├── models/vlas/
│   │   ├── openvla.py
│   │   ├── pi05_for_univla.py    # PI05 VLA implementation (updated)
│   │   └── __init__.py
│   ├── conf/vla.py
│   │   └── ...                    # Contains Exp_PI05_224px_Bridge config
│   └── load.py                   # Contains load_pi05() function
└── PI05_INTEGRATION_SUMMARY.md   # This file
```

## Notes

1. The PI05 model uses the same dataset pipeline as OpenVLA with OpenVLA action token mapping (tokens > 31744).
2. The `freeze_backbones()` method in PI05 properly freezes vision and LLM backbones during pretraining.
3. All scripts use `conda activate embodx` as specified in requirements.
