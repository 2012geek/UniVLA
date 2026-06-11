#!/usr/bin/env python3
"""
PI05 Pretraining on Bridge Dataset

This script pretrains PI05 on Bridge dataset using the UniVLA training pipeline.
The model learns to output LAM tokens as "high-order understanding" before
finetuning on task-specific data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python vla-scripts/train_pi05_bridge.py

Or with torchrun for multi-GPU:
    torchrun --nproc_per_node=1 vla-scripts/train_pi05_bridge.py
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import yaml

# Setup paths
sys.path.insert(0, '/root/autodl-tmp/workspace/fangziyu')
sys.path.insert(0, '/root/autodl-tmp/workspace/fangziyu/UniVLA')

# FIX: Set TORCH_HUB_DIR to use cached DINOv2
os.environ['TORCH_HUB_DIR'] = '/home/nice/.cache/torch/hub'

# Setup transformers check for PI05
import types
check_module = types.ModuleType('check')
check_module.check_whether_transformers_replace_is_installed_correctly = lambda: True
import transformers.models.siglip
transformers.models.siglip.check = check_module
sys.modules['transformers.models.siglip.check'] = check_module

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.conf.vla import Exp_PI05_224px_Bridge
from prismatic.models.load import load_pi05
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Patch torch.hub.load to use local DINOv2 cache
original_hub_load = torch.hub.load
def patched_hub_load(repo_or_dir, model, *args, **kwargs):
    if 'dinov2' in repo_or_dir:
        # Load directly from local cache (no GitHub validation)
        local_dir = os.environ.get('TORCH_HUB_DIR', '/home/nice/.cache/torch/hub')
        local_repo = os.path.join(local_dir, 'facebookresearch_dinov2_main')
        # Import directly from local path
        import sys
        sys.path.insert(0, local_repo)
        from dinov2.hub.backbones import dinov2_vitb14_reg
        return dinov2_vitb14_reg()
    return original_hub_load(repo_or_dir, model, *args, **kwargs)
torch.hub.load = patched_hub_load


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig - use PI05 Bridge config as template
    vla: Exp_PI05_224px_Bridge = field(
        default_factory=lambda: Exp_PI05_224px_Bridge(
            vla_id="pi05-bridge-pretrain",
            base_vlm="pi05-2b",
            data_mix="bridge",
            shuffle_buffer_size=10000,
            epochs=1,
            max_steps=100000,
            global_batch_size=1,
            per_device_batch_size=1,
            train_strategy="fsdp-shard-grad-op",
            expected_world_size=1,
            learning_rate=3e-4,
            weight_decay=0.05,
            max_grad_norm=1.0,
            lr_scheduler_type="linear-warmup+cosine-decay",
            warmup_ratio=0.1,
            enable_mixed_precision_training=False,  # Disabled - model is pure bfloat16
            enable_gradient_checkpointing=True,
        )
    )

    # PI05 specific settings
    lam_checkpoint_path: str = "/root/autodl-tmp/workspace/fangziyu/ckpt/univla-latent-action-model/lam-stage-2.ckpt"
    lam_vocab_size: int = 512
    lam_num_tokens: int = 4
    use_multi_token_prediction: bool = True

    # LAM settings (for creating action tokenizer)
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    codebook_size: int = 16

    # Directory Paths
    data_root_dir: Path = Path("/root/autodl-tmp/workspace/fangziyu/data")
    run_root_dir: Path = Path("/root/autodl-tmp/workspace/fangziyu/UniVLA/runs")

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None
    is_resume: bool = False
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None

    # Run Arguments
    run_id: Optional[str] = None
    run_id_note: Optional[str] = None
    save_interval: int = 100
    image_aug: bool = True
    seed: int = 42

    # HF Hub Credentials
    hf_token: Union[str, Path] = ''

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl",)
    wandb_project: str = "pi05-bridge-pretrain"
    wandb_entity: str = "pi05"

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("PI05 Bridge Pretraining :: Warming Up")

    # Setup torch.distributed
    # Handle both DistributedOverwatch and PureOverwatch
    if hasattr(overwatch, 'local_rank'):
        device_id = overwatch.local_rank()
    else:
        device_id = 0  # Single GPU training

    # Initialize distributed process group for FSDP (even for single GPU)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29501',
            world_size=1,
            rank=0
        )

    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.base_vlm
    cfg.run_id = (
        f"pi05-bridge-pretrain+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration
    if hasattr(overwatch, 'is_rank_zero') and overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load PI05 VLA
    overwatch.info(f"Loading PI05 VLM from scratch")
    vla = load_pi05(
        lam_vocab_size=cfg.lam_vocab_size,
        lam_num_tokens=cfg.lam_num_tokens,
        use_multi_token_prediction=cfg.use_multi_token_prediction,
        lam_checkpoint_path=cfg.lam_checkpoint_path,
    )

    # Convert ALL parameters to bfloat16 before FSDP wrapping
    # This includes LayerNorm, Embedding, etc. - everything must be uniform dtype
    overwatch.info(f"Converting model to pure bfloat16 (all layers)")
    vla = vla.to(torch.bfloat16).to(device_id)

    # Verify all params are bfloat16
    dtypes = set(p.dtype for p in vla.parameters())
    overwatch.info(f"Model parameter dtypes: {dtypes}")

    # All parameters are trainable (no freezing)
    # Train full model: vision_encoder, language_model, projector, sequence_head
    trainable_params = [name for name, param in vla.named_parameters() if param.requires_grad]
    overwatch.info(f"Trainable {len(trainable_params)} parameters (full model training)")

    # Print memory usage after model loading
    def print_memory_usage(stage: str) -> None:
        """Print GPU memory usage at different stages."""
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        overwatch.info(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    print_memory_usage("After model load to bfloat16")

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # Load LAM for dataset encoding (if not already loaded in PI05)
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    # Load the LAM model from checkpoint without using torch.hub
    ckpt = torch.load(cfg.lam_checkpoint_path)['state_dict']
    lam_state = {k.replace('lam.', ''): v for k, v in ckpt.items() if k.startswith('lam.')}

    # Initialize LAM model components from checkpoint state dict
    # This avoids the torch.hub.load() call that tries to download DINOv2
    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    # Load state from checkpoint
    latent_action_model.load_state_dict(lam_state, strict=True)

    # LAM model is only used for encoding, NOT training
    # Keep in float32 because DINO encoder + Normalize require float32
    # No gradients needed, so minimal memory overhead during forward
    # IMPORTANT: Keep LAM on CPU to save GPU memory! BatchTransform will move data to CPU for encoding.
    # latent_action_model = latent_action_model.to(device_id)  # DON'T move to GPU!
    latent_action_model = latent_action_model.eval()

    overwatch.info(f"LAM model kept on CPU with float32 precision (no gradients)")
    overwatch.info(f"This saves significant GPU memory - encoding happens on CPU")

    print_memory_usage("After LAM model load")

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Bridge Dataset")

    # LAM model stays in float32, so use default ToTensor transform (float32)
    image_transform_lam = transforms.ToTensor()

    vla_dataset, action_tokenizer, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,  # "bridge" or "bridge_dataset"
        image_transform=vla.vision_backbone.get_image_transform(),
        image_transform_lam=image_transform_lam,
        latent_action_tokenizer=latent_action_model,
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=False,  # Disabled for Bridge pretraining
    )

    overwatch.info(f"Dataset size: {len(vla_dataset)}")

    # Add special tokens for LAM
    # Note: SimpleTokenizer doesn't have add_special_tokens, skip for now
    # special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    # num_added_toks = action_tokenizer.add_special_tokens(special_tokens_dict)

    # Save dataset statistics for de-normalization at inference time
    if not hasattr(overwatch, 'is_rank_zero') or overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage="vla-train",  # Frozen vision + LLM, train adapter
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    print_memory_usage("After FSDP setup")

    # Create Metrics
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting PI05 Bridge Pretraining")
    train_strategy.run_latent_action_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    # Only cleanup distributed if it was initialized
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
