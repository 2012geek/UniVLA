"""
PI05-adapted Fine-tuning Script for LIBERO

This script is adapted from finetune_libero.py to support both PI05 and OpenVLA models.

Key Changes for PI05:
    - Added `use_pi05` flag and related config parameters
    - Conditional model loading (PI05 vs OpenVLA)
    - Modified forward pass for PI05
    - OpenVLA action token mapping for loss computation
    - Different accuracy computation for PI05's multi-token format

Architecture (PI05):
    - PI05 (PaliGemma-2B + Gemma-300M) - frozen
    - LAM (VQ-VAE) - frozen
    - Projector + Sequence Head - trainable (~34.6M params)
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformLIBERO, RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prismatic.models.policy.transformer_utils import MAPBlock

# CHANGE [PI05]: Import PI05 loader
from prismatic.models.load import load_pi05

# =============================================================================
# Action Decoder (shared between OpenVLA and PI05)
# =============================================================================

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        # Split output: first 6 dims (position+rotation) with Tanh, 7th dim (gripper) with Sigmoid
        self.proj_pos_rot = nn.Linear(hidden_dim, 6 * window_size)
        self.proj_gripper = nn.Linear(hidden_dim, 1 * window_size)

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        # First 6 dims: use Tanh -> [-1, 1] (for normalized pos/rot)
        pos_rot = torch.tanh(self.proj_pos_rot(action_token))

        # Gripper: use Sigmoid -> [0, 1] (directly matches GT)
        gripper = torch.sigmoid(self.proj_gripper(action_token))

        # Concatenate
        action = torch.cat([pos_rot, gripper], dim=-1)

        return action


# =============================================================================
# Wrapped Model (supports both OpenVLA and PI05)
# =============================================================================

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, use_pi05=False, freeze_vla=False, window_size=12,
                 pi05_lam_vocab_size=512, pi05_lam_num_tokens=4):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.use_pi05 = use_pi05
        self.pi05_lam_vocab_size = pi05_lam_vocab_size
        self.pi05_lam_num_tokens = pi05_lam_num_tokens
        self.action_decoder = ActionDecoder(window_size=window_size)

        # CHANGE [PI05]: For PI05, don't set requires_grad on entire model
        # freeze_backbones already handles it correctly for PI05
        if freeze_vla and not self.use_pi05:
            self.vla.requires_grad_(False)

        # CHANGE [PI05]: Pre-create projection layers in __init__ for PI05
        # This ensures they are registered as model parameters and included in optimizer
        if use_pi05:
            llm_dim = 2048  # Gemma-2B hidden size
            self.latent_proj_pi05 = nn.Linear(llm_dim, 4096)
            self.visual_proj_pi05 = nn.Linear(llm_dim, 4096)

    def forward(self, batch):
        if self.use_pi05:
            return self.forward_pi05(batch)
        else:
            return self.forward_openvla(batch)

    def forward_openvla(self, batch):
        """Forward pass for OpenVLA model."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)
        return vla_output, loss, loss_one_step, latent_action_tokens

    def forward_pi05(self, batch):
        """Forward pass for PI05 model with token mapping."""
        # PI05 forward pass - get hidden states for action decoder
        # NOTE: PI05 doesn't use LAM token labels like OpenVLA, so we pass labels=None
        # The loss is computed only on the action decoder side
        vla_output = self.vla(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=None,  # PI05 doesn't predict LAM tokens, only train action decoder
            output_hidden_states=True,  # Need hidden states for action decoder
        )

        # For PI05, vla_loss is 0 (we only train action decoder)
        vla_loss = torch.tensor(0.0, device=batch["pixel_values"].device)

        # Extract hidden states for action decoder training
        hidden_states = vla_output.hidden_states[-1]  # [B, seq_len, hidden_dim]

        # Get visual embedding (vision tokens)
        with torch.no_grad():
            image_features = self.vla.encode_image(batch["pixel_values"])
        num_patches = image_features.shape[1]

        # Visual embed: use projected vision features
        visual_embed = self.vla.projector(image_features)  # [B, num_patches, llm_embed_dim]
        # Pool to single vector
        visual_embed = visual_embed.mean(dim=1, keepdim=True)  # [B, 1, llm_embed_dim]

        # Latent action tokens: extract text tokens after vision patches
        text_hidden_states = hidden_states[:, num_patches:, :]  # [B, text_seq_len, hidden_dim]
        # Get last 4 tokens as latent action (matching decoder expectation)
        latent_action_tokens = text_hidden_states[:, -4:, :]  # [B, 4, hidden_dim]

        # Project to 4096 dim (action decoder expects 4096)
        # Projection layers are pre-created in __init__ and will be trained by optimizer
        latent_action_tokens = self.latent_proj_pi05(latent_action_tokens)
        visual_embed = self.visual_proj_pi05(visual_embed)

        # Compute action decoder loss
        # ActionDecoder now outputs: [0:6] -> Tanh [-1,1], [6] -> Sigmoid [0,1]
        # Gripper directly matches GT [0,1], no mapping needed
        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        action_loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = action_loss[:, 0].mean()
        action_loss = action_loss.mean()

        # Combine VLA token loss and action decoder loss
        loss = vla_loss + action_loss

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        """Compute action decoder loss (for OpenVLA)."""
        visual_embed = vla_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches ].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)

        # ActionDecoder now outputs: [0:6] -> Tanh [-1,1], [6] -> Sigmoid [0,1]
        # Gripper directly matches GT [0,1], no mapping needed
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FinetuneConfig:
    # fmt: off
    # Model selection
    # CHANGE [PI05]: Added use_pi05 flag
    use_pi05: bool = False                                          # Use PI05 model
    pi05_lam_vocab_size: int = 512                                 # LAM vocab size
    pi05_lam_num_tokens: int = 4                                   # Number of LAM tokens
    pi05_use_multi_token_prediction: bool = True                   # Multi-token prediction
    pi05_pretrained_checkpoint: Optional[str] = None               # Path to pretrained PI05 checkpoint (.pt)

    vla_path: str = "/path/to/your/pretrained-univla-7b"            # Path to your local UniVLA path
    lam_path: str = "/root/autodl-tmp/workspace/hmx/ckpt/univla-latent-action-model/lam-stage-2.ckpt"

    # Directory Paths
    data_root_dir: Path = Path("/LIBERO/modified_libero_rlds")      # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                   # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 30000                                          # Max number of fine-tuning steps
    save_steps: int = 30000                                         # Interval for checkpoint saving
    learning_rate: float = 3.5e-4                                   # Fine-tuning learning rate
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 16000                                # Dataloader shuffle buffer size
    save_latest_checkpoint_only: bool = True                        # Save only latest checkpoint

    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    # LoRA Arguments (not used for PI05)
    freeze_vla: bool = False
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA

    # Tracking Parameters
    wandb_project: str = "finetune-LIBERO"                         # Name of W&B project
    wandb_entity: str = "opendrivelab"                             # Name of entity to log under
    run_id_note: Optional[str] = None                              # Extra note for logging

    # PyTorch Compile Mode (for speed & memory optimization)
    compile_mode: str = "max-autotune-no-cudagraphs"                # "default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead", or None to disable


# =============================================================================
# Main Fine-tuning Function
# =============================================================================

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Print model type
    if cfg.use_pi05:
        print(f"Fine-tuning PI05 Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    else:
        print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    if cfg.use_pi05:
        exp_id = f"PI05+{cfg.dataset_name}"
    else:
        exp_id = f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"

    exp_id += f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    exp_id += f"+lr-{cfg.learning_rate}"

    if cfg.use_lora and not cfg.use_pi05:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # CHANGE [PI05]: Conditional model loading
    if cfg.use_pi05:
        # Load PI05 model
        print("[PI05] Loading PI05 model...")
        vla = load_pi05(
            model_id=cfg.vla_path,
            lam_vocab_size=cfg.pi05_lam_vocab_size,
            lam_num_tokens=cfg.pi05_lam_num_tokens,
            lam_checkpoint_path=cfg.lam_path,
            use_multi_token_prediction=cfg.pi05_use_multi_token_prediction,
            enable_mixed_precision_training=True,
            pretrained_checkpoint=cfg.pi05_pretrained_checkpoint,
        )
        vla.freeze_backbones("align")

        # Load LAM model for dataset transformation
        # Set proxy and TORCH_HOME for downloading DINOv2
        os.environ['http_proxy'] = 'http://172.32.52.144:12798'
        os.environ['https_proxy'] = 'http://172.32.52.144:12798'
        os.environ['TORCH_HOME'] = '/home/nice/.cache/torch'
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
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
        lam_ckpt = torch.load(cfg.lam_path)['state_dict']
        new_ckpt = {}
        for key in lam_ckpt.keys():
            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
        latent_action_model.load_state_dict(new_ckpt, strict=True)
        # Keep LAM on CPU to save GPU memory during data loading
        latent_action_model = latent_action_model.to('cpu').eval()

        # CHANGE [PI05]: Create proper image transform for PI05
        pi05_image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # CHANGE [PI05]: Use simple tokenizer placeholder for PI05
        class SimpleTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
            def __call__(self, *args, **kwargs):
                return {"input_ids": kwargs.get("input_ids", [])}

        processor = type('Processor', (), {
            'tokenizer': SimpleTokenizer(),
            'image_processor': type('ImageProcessor', (), {
                'apply_transform': lambda x: x
            })()
        })()

        # For PI05, we don't use OpenVLA's action tokenizer
        action_tokenizer = None

        # Create PI05 batch transform
        batch_transform = RLDSBatchTransformLIBERO_withHis(
            latent_action_model,
            processor.tokenizer,
            image_transform=pi05_image_transform,  # Use PI05's transform
            image_transform_lam=transforms.ToTensor(),
            prompt_builder_fn=PurePromptBuilder,
            window_size=cfg.window_size
        )

        print("[PI05] Model loaded. LAM vocab size: {}, LAM num tokens: {}".format(
            cfg.pi05_lam_vocab_size, cfg.pi05_lam_num_tokens))

    else:
        # Load OpenVLA model (original code)
        quantization_config = None
        if cfg.use_quantization:
            assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if cfg.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        else:
            vla = vla.to(device_id)

        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

        action_tokenizer = ActionTokenizer(processor.tokenizer)

        # Load LAM model for dataset transformation
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
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
        lam_ckpt = torch.load(cfg.lam_path)['state_dict']
        new_ckpt = {}
        for key in lam_ckpt.keys():
            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
        latent_action_model.load_state_dict(new_ckpt, strict=True)
        # Keep LAM on CPU to save GPU memory during data loading
        latent_action_model = latent_action_model.to('cpu').eval()

        batch_transform = RLDSBatchTransformLIBERO_withHis(
            latent_action_model,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            image_transform_lam=transforms.ToTensor(),
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
            window_size=cfg.window_size
        )

    # Wrap model on CPU first (before torch.compile to save memory)
    print("[DEBUG] Creating wrapped model on CPU...")
    wrapped_model = Wrapped_Model(
        vla=vla,
        use_pi05=cfg.use_pi05,
        freeze_vla=cfg.freeze_vla,
        window_size=cfg.window_size,
        pi05_lam_vocab_size=cfg.pi05_lam_vocab_size,
        pi05_lam_num_tokens=cfg.pi05_lam_num_tokens,
    )
    # Keep model on CPU for torch.compile

    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    # Apply torch.compile on CPU (before moving to GPU) to save memory
    if cfg.compile_mode is not None and cfg.compile_mode.lower() != "none":
        print(f'[DEBUG] Applying torch.compile with mode="{cfg.compile_mode}" on CPU before moving to GPU')
        wrapped_model = torch.compile(wrapped_model, mode=cfg.compile_mode)

    # Now move to GPU after compilation
    print(f'[DEBUG] Moving model to GPU {device_id}...')
    wrapped_model = wrapped_model.to(device_id)
    torch.cuda.empty_cache()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    if torch.cuda.device_count() > 1:
        wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
        print(f'Using DDP with {torch.cuda.device_count()} GPUs')
        use_ddp = True
    else:
        print('Single GPU training - skipping DDP wrapper')
        use_ddp = False

    def get_model(m):
        return m.module if isinstance(m, DDP) else m

    # Create Optimizer
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)

    # Create Dataset
    print(f"[DEBUG] Creating dataset from: {cfg.data_root_dir}, dataset_name: {cfg.dataset_name}")
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(224, 224) if cfg.use_pi05 else tuple(get_model(wrapped_model).vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,
        training_phase='post-training',
    )
    print(f"[DEBUG] Dataset created successfully")
    print(f"[DEBUG] Dataset statistics: {vla_dataset.dataset_statistics}")

    # Save Dataset Statistics
    if not use_ddp or distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # CHANGE [PI05]: Use 512 for max length (same as original)
    collator = PaddedCollatorForActionPrediction_LIBERO(
        model_max_length=512,
        pad_token_id=0,
        padding_side="right"
    )

    print(f"[DEBUG] Creating dataloader with batch_size={cfg.batch_size}, num_workers=0")
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle
        collate_fn=collator,
        num_workers=0,  # Single worker for clearer debugging
        pin_memory=True,
        drop_last=True,
    )
    print(f"[DEBUG] Dataloader created successfully")

    # W&B Logging - skip if disabled
    wandb_run = None
    if not use_ddp or distributed_state.is_main_process:
        if cfg.wandb_project != "disabled":
            try:
                # Use offline mode if WANDB_MODE is set
                mode = os.environ.get("WANDB_MODE", "online" if cfg.run_id_note != "debug" else "disabled")
                wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=exp_id,
                    config=cfg.__dict__,
                    mode=mode,
                )
                wandb_run = wandb
            except Exception as e:
                print(f"Warning: W&B initialization failed: {e}")
                print("Continuing training without W&B logging...")
                wandb_run = None

    # Training Loop
    wrapped_model.train()
    optimizer.zero_grad()

    cumulative_loss = deque(maxlen=100)
    gradient_step_idx = 0

    pbar = tqdm.tqdm(total=cfg.max_steps, desc="Fine-tuning", disable=not (not use_ddp or distributed_state.is_main_process))

    print(f"[DEBUG] Starting training loop, about to load first batch...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"[DEBUG] First batch loaded successfully! batch_idx={batch_idx}")
        if gradient_step_idx >= cfg.max_steps:
            break

        # Move batch to device
        batch = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        vla_output, loss, loss_one_step, latent_action_tokens = wrapped_model(batch)

        # Backward pass
        loss = loss / cfg.grad_accumulation_steps
        loss.backward()

        cumulative_loss.append(loss.item() * cfg.grad_accumulation_steps)

        # Optimizer step
        if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            gradient_step_idx += 1

            # Logging
            if not use_ddp or distributed_state.is_main_process:
                if wandb_run is not None:
                    wandb.log({
                        "train_loss": sum(cumulative_loss) / len(cumulative_loss),
                        "action_loss": loss.item() * cfg.grad_accumulation_steps,
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=gradient_step_idx)

                # Write all losses to a single log file
                with open(f"{run_dir}/training_log.txt", "a") as f:
                    f.write(f"Step {gradient_step_idx}: train_loss={sum(cumulative_loss) / len(cumulative_loss):.6f}, action_loss={loss.item() * cfg.grad_accumulation_steps:.6f}, action_loss_1step={loss_one_step.item():.6f}, lr={optimizer.param_groups[0]['lr']:.6e}\n")

                pbar.set_description(
                    f"[Step {gradient_step_idx}] train_loss={sum(cumulative_loss) / len(cumulative_loss):.4f}, "
                    f"action_loss={loss.item() * cfg.grad_accumulation_steps:.4f}, "
                    f"action_loss_1step={loss_one_step.item():.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            pbar.update(1)

            # Save checkpoint
            if gradient_step_idx % cfg.save_steps == 0:
                if not use_ddp or distributed_state.is_main_process:
                    save_dir = adapter_dir if cfg.use_lora and not cfg.freeze_vla else run_dir

                    # CHANGE [PI05]: Save logic for PI05 (no LoRA, simpler save)
                    if cfg.use_pi05:
                        # FIX: Use _use_new_zipfile_serialization=False to avoid filesystem errors
                        torch.save({
                            'step': gradient_step_idx,
                            'model_state_dict': get_model(wrapped_model).vla.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, str(run_dir) + f'/pi05-checkpoint-{gradient_step_idx}.pt',
                        _use_new_zipfile_serialization=False)
                        print(f"Saved PI05 Checkpoint at: {run_dir}/pi05-checkpoint-{gradient_step_idx}.pt")
                    elif not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        get_model(wrapped_model).vla.save_pretrained(save_dir)

                    # Save low-level policy (including projection layers for PI05)
                    action_decoder_state = get_model(wrapped_model).action_decoder.state_dict()
                    # Save projection layers if they exist (for PI05)
                    # Check for both old and new key names for backward compatibility
                    if hasattr(get_model(wrapped_model), 'latent_proj_pi05'):
                        action_decoder_state['latent_proj_pi05'] = get_model(wrapped_model).latent_proj_pi05.state_dict()
                    elif hasattr(get_model(wrapped_model), 'latent_proj'):
                        action_decoder_state['latent_proj'] = get_model(wrapped_model).latent_proj.state_dict()
                    if hasattr(get_model(wrapped_model), 'visual_proj_pi05'):
                        action_decoder_state['visual_proj_pi05'] = get_model(wrapped_model).visual_proj_pi05.state_dict()
                    elif hasattr(get_model(wrapped_model), 'visual_proj'):
                        action_decoder_state['visual_proj'] = get_model(wrapped_model).visual_proj.state_dict()
                    torch.save(action_decoder_state,
                              str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt')

                if use_ddp:
                    dist.barrier()

            if gradient_step_idx >= cfg.max_steps - 1:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    # Save final checkpoint
    if not use_ddp or distributed_state.is_main_process:
        if cfg.use_pi05:
            # FIX: Use _use_new_zipfile_serialization=False
            torch.save({
                'step': cfg.max_steps,
                'model_state_dict': get_model(wrapped_model).vla.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(run_dir) + '/pi05-final.pt',
            _use_new_zipfile_serialization=False)
            print(f"Saved final PI05 checkpoint at: {run_dir}/pi05-final.pt")
        elif not cfg.freeze_vla:
            processor.save_pretrained(run_dir)
            get_model(wrapped_model).vla.save_pretrained(run_dir)


if __name__ == "__main__":
    finetune()
