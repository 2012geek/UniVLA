#!/bin/bash
################################################################################
# Fine-tuning Script for EmbodX Baseline (PI05 version)
#
# This script fine-tunes the pretrained VLA model on LIBERO dataset.
# Now supports both OpenVLA and PI05 models via command-line flags.
#
# CHANGE [PI05]: Added use_pi05 flag and related parameters
################################################################################
set -e

# Path configuration
VLA_PATH="/root/autodl-tmp/workspace/hmx/ckpt/univla-7b-bridge-pt"
DATA_ROOT_DIR="/root/autodl-tmp/workspace/hmx/data"
RUN_DIR="/root/autodl-tmp/workspace/fangziyu/UniVLA/runs/PI05+libero_spatial_no_noops+b64+lr-0.00035--end2end--image_aug=w-LowLevelDecoder-ws-12"
ADAPTER_TMP_DIR="adapter-tmp"

# Training configuration
DATASET_NAME="libero_spatial_no_noops"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=8  # Effective batch = 8 * 8 = 64
LEARNING_RATE=1e-4  # Fine-tuning learning rate
MAX_STEPS=30000
SAVE_STEPS=3000
LORA_RANK=32
LORA_DROPOUT=0.0
WINDOW_SIZE=12
IMAGE_AUG=True
USE_QUANTIZATION=False

# LoRA configuration
USE_LORA=False  # End-to-end training, no LoRA
FREEZE_VLA=False  # End-to-end, VLM not frozen

# CHANGE [PI05]: PI05-specific parameters
USE_PI05=true                         # Use PI05 model instead of OpenVLA
PI05_LAM_VOCAB_SIZE=512             # LAM vocab size
PI05_LAM_NUM_TOKENS=4               # Number of LAM tokens
PI05_USE_MULTI_TOKEN_PREDICTION=true  # Multi-token prediction
# Correct path to pretrained PI05 checkpoint (from bridge pretraining)
PI05_PRETRAINED_CHECKPOINT="/root/autodl-tmp/workspace/hmx/UniVLA/runs/pi05-bridge-pretrain+b1+x42/checkpoints/step-002500-epoch-00-loss=0.0000.pt"

# PyTorch Compile Mode (for speed & memory optimization)
COMPILE_MODE="max-autotune-no-cudagraphs"  # "default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead", or "none" to disable

# WandB configuration
WANDB_PROJECT="finetune-LIBERO"
WANDB_ENTITY="opendrivelab"
RUN_ID_NOTE="end2end"

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_pi05)
            USE_PI05=true
            ;;
        --pi05_lam_vocab_size=*)
            PI05_LAM_VOCAB_SIZE="${1#*=}"
            ;;
        --pi05_lam_num_tokens=*)
            PI05_LAM_NUM_TOKENS="${1#*=}"
            ;;
        --pi05_use_multi_token_prediction=*)
            PI05_USE_MULTI_TOKEN_PREDICTION="${1#*=}"
            ;;
        --pi05_pretrained_checkpoint=*)
            PI05_PRETRAINED_CHECKPOINT="${1#*=}"
            ;;
        --vla_path=*)
            VLA_PATH="${1#*=}"
            ;;
        --max_steps=*)
            MAX_STEPS="${1#*=}"
            ;;
        --save_steps=*)
            SAVE_STEPS="${1#*=}"
            ;;
        --learning_rate=*)
            LEARNING_RATE="${1#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            ;;
        --grad_accum_steps=*)
            GRAD_ACCUM_STEPS="${1#*=}"
            ;;
        --dataset_name=*)
            DATASET_NAME="${1#*=}"
            ;;
        --data_root_dir=*)
            DATA_ROOT_DIR="${1#*=}"
            ;;
        --window_size=*)
            WINDOW_SIZE="${1#*=}"
            ;;
        --compile_mode=*)
            COMPILE_MODE="${1#*=}"
            ;;
    esac
    shift
done

echo "========================================="
echo "Fine-tuning for EmbodX Baseline (PI05)"
echo "========================================="
echo "VLA Path: $VLA_PATH"
echo "Dataset: $DATASET_NAME"
echo "Batch Size: $BATCH_SIZE (Effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Steps: $MAX_STEPS"
echo "LoRA Rank: $LORA_RANK"
echo "Freeze VLA: $FREEZE_VLA"
echo "Use PI05: $USE_PI05"
echo "Compile Mode: $COMPILE_MODE"
if [ -n "$PI05_PRETRAINED_CHECKPOINT" ]; then
    echo "PI05 Pretrained Checkpoint: $PI05_PRETRAINED_CHECKPOINT"
fi
echo "========================================="

python3 vla-scripts/finetune_libero_pi05.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_DIR" \
    --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --learning_rate $LEARNING_RATE \
    --grad_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_aug $IMAGE_AUG \
    --shuffle_buffer_size 100 \
    --save_latest_checkpoint_only False \
    --window_size $WINDOW_SIZE \
    --freeze_vla $FREEZE_VLA \
    --use_lora $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_dropout $LORA_DROPOUT \
    --use_quantization $USE_QUANTIZATION \
    --compile_mode "$COMPILE_MODE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_pi05 $USE_PI05 \
    --pi05_lam_vocab_size $PI05_LAM_VOCAB_SIZE \
    --pi05_lam_num_tokens $PI05_LAM_NUM_TOKENS \
    --pi05_use_multi_token_prediction $PI05_USE_MULTI_TOKEN_PREDICTION \
    --pi05_pretrained_checkpoint "$PI05_PRETRAINED_CHECKPOINT"

echo ""
echo "Fine-tuning completed!"
echo "Checkpoints saved in: $RUN_DIR"
