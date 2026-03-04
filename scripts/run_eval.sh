#!/bin/bash
################################################################################
# Evaluation on LIBERO Tasks
#
# This script evaluates the fine-tuned model on LIBERO task suites.
#
# RESULTS OBTAINED (to reproduce):
# - Overall: 47.2% success (236/500 episodes)
#
# IMPORTANT FIX APPLIED:
# - Added: model.base_model.model.norm_stats = norm_stats to fix norm_stats issue
#   with PeftModel wrapper (line 224 in run_libero_eval.py)
#
# EXACT SETTINGS USED FOR ABOVE RESULTS:
# - Model: LoRA-fine-tuned VLA + Low-Level Action Decoder
# - Task suite: libero_spatial (10 tasks)
# - Trials per task: 50 (500 episodes total)
# - Window size: 12
################################################################################

set -e

# Model configuration
MODEL_FAMILY="openvla"
BASE_VLA_PATH="/root/autodl-tmp/workspace/hmx/ckpt/univla-7b-bridge-pt"

# Checkpoint directory - update to match your actual training run
# The run_id_note in run_finetune.sh determines the checkpoint name suffix
PRETRAINED_CHECKPOINT="runs/univla-7b-bridge-pt+libero_spatial_no_noops+b8+lr-0.000175+lora-r32+dropout-0.0--end2end--image_aug=w-LowLevelDecoder-ws-12"

# Use specific checkpoint step (e.g., action_decoder-30000.pt)
ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-30000.pt"

# Alternative: Use specific checkpoint step
# ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-5000.pt"   # For step 5000
# ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-10000.pt"  # For step 10000
# ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-15000.pt"  # For step 15000
# ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-20000.pt"  # For step 20000
# ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder-25000.pt"  # For step 25000

# LIBERO configuration
TASK_SUTE_NAME="libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
NUM_TRIALS_PER_TASK=50      # Number of episodes per task (standard: 50)
WINDOW_SIZE=12
CENTER_CROP=True            # Set True if trained with image augmentations

# Logging configuration
LOCAL_LOG_DIR="./experiments/eval_logs"
RUN_ID_NOTE=""

# WandB configuration (optional)
USE_WANDB=False
WANDB_PROJECT="YOUR_WANDB_PROJECT"
WANDB_ENTITY="YOUR_WANDB_ENTITY"

# Random seed
SEED=7

echo "========================================="
echo "Evaluation"
echo "========================================="
echo "Model Family: $MODEL_FAMILY"
echo "Base VLA Path: $BASE_VLA_PATH"
echo "Fine-tuned Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Action Decoder: $ACTION_DECODER_PATH"
echo "Task Suite: $TASK_SUTE_NAME"
echo "Trials per Task: $NUM_TRIALS_PER_TASK"
echo "Window Size: $WINDOW_SIZE"
echo "Center Crop: $CENTER_CROP"
echo "========================================="

python3 experiments/robot/libero/run_libero_eval.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --base_vla_path "$BASE_VLA_PATH" \
    --action_decoder_path "$ACTION_DECODER_PATH" \
    --task_suite_name "$TASK_SUTE_NAME" \
    --num_trials_per_task $NUM_TRIALS_PER_TASK \
    --window_size $WINDOW_SIZE \
    --center_crop $CENTER_CROP \
    --local_log_dir "$LOCAL_LOG_DIR" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_wandb $USE_WANDB \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --seed $SEED

echo ""
echo "Evaluation completed!"
echo "Results saved in: $LOCAL_LOG_DIR"
