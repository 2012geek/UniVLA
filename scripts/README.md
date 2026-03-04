# UniVLA Fine-tuning and Evaluation Scripts

This directory contains scripts for fine-tuning and evaluating UniVLA models on LIBERO tasks.

---

## Fine-tuning Script: `run_step1_finetune.sh`

This script fine-tunes the pretrained VLA model on LIBERO dataset using LoRA with all parameters trainable.

### Results Obtained

| Metric | Value |
|--------|-------|
| Training Steps | 30000/30000 (completed successfully) |
| Training Loss | 1.26 -> 0.66 |
| Training Accuracy | 56% -> 81% |
| Checkpoints Saved | 5000, 10000, 15000, 20000, 25000, 30000 |
| Evaluation Overall | **47.2% success** (236/500 episodes) |

### Exact Configuration Used

```bash
VLA_PATH="/root/autodl-tmp/workspace/hmx/ckpt/univla-7b-bridge-pt"
DATASET_NAME="libero_spatial_no_noops"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=8  # Effective batch = 64
LEARNING_RATE=1.75e-4  # SCALED from 3.5e-4 for batch size 8 vs 16
MAX_STEPS=30000
SAVE_STEPS=5000
LORA_RANK=32
LORA_DROPOUT=0.0
WINDOW_SIZE=12
IMAGE_AUG=True
FREEZE_VLA=False  # End-to-end, VLM not frozen
```

### To Run Fine-tuning

```bash
./scripts/run_step1_finetune.sh
```

---

## Evaluation Script: `run_step1_eval.sh`

This script evaluates the fine-tuned model on LIBERO task suites.

### Results Obtained

**Overall: 47.2% success rate** (236/500 episodes)

| Task | Description | Success Rate | Successes/50 |
|------|-------------|--------------|--------------|
| Task 1 | pick up black bowl between plate and ramekin | **68.0%** | 34/50 |
| Task 2 | pick up black bowl next to ramekin | **68.0%** | 34/50 |
| Task 3 | pick up black bowl from table center | **34.0%** | 17/50 |
| Task 4 | pick up black bowl on cookie box | **98.0%** | 49/50 |
| Task 5 | pick up black bowl in top drawer of wooden cabinet | **38.0%** | 19/50 |
| Task 6 | pick up black bowl on ramekin | **24.0%** | 12/50 |
| Task 7 | pick up black bowl next to cookie box | **20.0%** | 10/50 |
| Task 8 | pick up black bowl on stove | **74.0%** | 37/50 |
| Task 9 | pick up black bowl next to plate | **16.0%** | 8/50 |
| Task 10 | pick up black bowl on wooden cabinet | **32.0%** | 16/50 |

**Best performing task:** Task 4 (98.0% - pick up black bowl on cookie box)
**Worst performing task:** Task 9 (16.0% - pick up black bowl next to plate)

### Important Fix Applied

**File**: `experiments/robot/libero/run_libero_eval.py`

When using LoRA-fine-tuned models, the `norm_stats` attribute needs to be set on the underlying base model, not just the PeftModel wrapper. This was fixed by adding:

```python
# Line 224 in run_libero_eval.py
model.base_model.model.norm_stats = norm_stats
```

Without this fix, evaluation would fail with:
```
The `unnorm_key` you chose is not in the set of available dataset statistics,
please choose from: dict_keys(['bridge_oxe'])
```

### Exact Configuration Used

```bash
MODEL_FAMILY="openvla"
BASE_VLA_PATH="/root/autodl-tmp/workspace/hmx/ckpt/univla-7b-bridge-pt"
PRETRAINED_CHECKPOINT="runs/univla-7b-bridge-pt+libero_spatial_no_noops+b8+lr-0.000175+lora-r32+dropout-0.0--step1-end2end--image_aug=w-LowLevelDecoder-ws-12"
ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-30000.pt"
TASK_SUTE_NAME="libero_spatial"
NUM_TRIALS_PER_TASK=50
WINDOW_SIZE=12
CENTER_CROP=True
```

### To Run Evaluation

```bash
./scripts/run_step1_eval.sh
```

To evaluate with a specific checkpoint (not 30000), edit the script:

```bash
# Use action_decoder-5000.pt, action_decoder-10000.pt, etc.
ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-5000.pt"
```

---

## Task Suites

The evaluation script supports multiple LIBERO task suites:

- `libero_spatial`: 10 tasks, spatial reasoning
- `libero_object`: 10 tasks, object manipulation
- `libero_goal`: 10 tasks, goal-conditioned
- `libero_10`: 10 tasks
- `libero_90`: 90 tasks (longer evaluation)

To change task suite, edit the `TASK_SUTE_NAME` variable in `run_step1_eval.sh`:

```bash
TASK_SUTE_NAME="libero_object"  # or libero_goal, libero_10, libero_90
```

---

## Files

| Script | Purpose |
|--------|---------|
| `run_step1_finetune.sh` | Fine-tune pretrained VLM on LIBERO |
| `run_step1_eval.sh` | Evaluate fine-tuned model |

---

## Notes

1. **Learning Rate Scaling**: For smaller batch sizes, scale the learning rate proportionally. The original used batch size 16 with lr=3.5e-4. For batch size 8, we used lr=1.75e-4.

2. **LoRA Configuration**: The LoRA rank (32) and dropout (0.0) were used for fine-tuning.

3. **Window Size**: Action window size of 12 was used for temporal action prediction.

4. **Image Augmentation**: Random crop augmentation was enabled during training (`image_aug=True`), so `center_crop=True` during evaluation.

5. **Gradient Accumulation**: Used to achieve larger effective batch sizes with smaller per-GPU batch size.

6. **Checkpoint Selection**: Multiple checkpoints were saved during training. You can evaluate any checkpoint to see how performance evolves with training progress.

7. **Norm Stats Fix**: When using LoRA-fine-tuned models for evaluation, ensure `model.base_model.model.norm_stats` is set to the correct dataset statistics (see the Important Fix section above).
