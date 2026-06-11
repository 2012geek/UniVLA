#!/bin/bash
################################################################################
# Evaluation Script for PI05 Model on LIBERO
#
# This script evaluates the PI05 fine-tuned model on LIBERO task suites.
#
# USAGE:
#   ./baseline_scripts/run_eval_pi05.sh --checkpoint_step 10000
################################################################################

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate embodxtest

set -e

# Default values
CHECKPOINT_STEP=3000
TASK_SUITE="libero_spatial"
NUM_TRIALS=10
WINDOW_SIZE=12
CENTER_CROP=False  # Disabled to match training

# Checkpoint directory (from fine-tuning)
CHECKPOINT_DIR="/root/autodl-tmp/workspace/fangziyu/UniVLA/runs/PI05+libero_spatial_no_noops+b64+lr-0.00035--end2end--image_aug=w-LowLevelDecoder-ws-12/PI05+libero_spatial_no_noops+b64+lr-0.0001--end2end--image_aug=w-LowLevelDecoder-ws-12"

# LAM checkpoint (using fangziyu path for consistency)
LAM_CHECKPOINT="/root/autodl-tmp/workspace/fangziyu/ckpt/univla-latent-action-model/lam-stage-2.ckpt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_step=*)
            CHECKPOINT_STEP="${1#*=}"
            ;;
        --task_suite=*)
            TASK_SUITE="${1#*=}"
            ;;
        --num_trials=*)
            NUM_TRIALS="${1#*=}"
            ;;
        --checkpoint_dir=*)
            CHECKPOINT_DIR="${1#*=}"
            ;;
        --window_size=*)
            WINDOW_SIZE="${1#*=}"
            ;;
        --center_crop=*)
            CENTER_CROP="${1#*=}"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Construct checkpoint paths
PI05_CHECKPOINT="${CHECKPOINT_DIR}/pi05-checkpoint-${CHECKPOINT_STEP}.pt"
ACTION_DECODER_PATH="${CHECKPOINT_DIR}/action_decoder-${CHECKPOINT_STEP}.pt"

# Verify checkpoints exist
if [ ! -f "$PI05_CHECKPOINT" ]; then
    echo "ERROR: PI05 checkpoint not found: $PI05_CHECKPOINT"
    exit 1
fi

if [ ! -f "$ACTION_DECODER_PATH" ]; then
    echo "ERROR: Action decoder checkpoint not found: $ACTION_DECODER_PATH"
    exit 1
fi

echo "========================================="
echo "PI05 Evaluation on LIBERO"
echo "========================================="
echo "PI05 Checkpoint: $PI05_CHECKPOINT"
echo "Action Decoder: $ACTION_DECODER_PATH"
echo "Task Suite: $TASK_SUITE"
echo "Trials per Task: $NUM_TRIALS"
echo "Window Size: $WINDOW_SIZE"
echo "Center Crop: $CENTER_CROP"
echo "========================================="

python3 experiments/robot/libero/run_libero_eval_pi05.py \
    --pi05_checkpoint "$PI05_CHECKPOINT" \
    --action_decoder_path "$ACTION_DECODER_PATH" \
    --lam_checkpoint_path "$LAM_CHECKPOINT" \
    --lam_vocab_size 512 \
    --lam_num_tokens 4 \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task $NUM_TRIALS \
    --window_size $WINDOW_SIZE \
    --center_crop $CENTER_CROP \
    --local_log_dir "experiments/eval_logs" \
    --run_id_note "pi05-step${CHECKPOINT_STEP}" \
    --seed 7 \
    | grep -v "gripper raw=" | grep -E "(Action:|distance|EEF|Success|Task:|===|\[.*\])"

echo ""
echo "Evaluation completed!"
echo "Results saved in: ./experiments/eval_logs"
