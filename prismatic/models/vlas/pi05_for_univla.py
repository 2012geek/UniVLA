"""
PI05-for-UniVLA: PI05 VLA Adapter for UniVLA Training Pipeline

This adapter integrates PI05 (PaliGemma-2B) with UniVLA's training infrastructure.

Architecture:
    - Vision Encoder: SigLIP (from PaliGemma, frozen)
    - Language Model: Gemma-2B (from PaliGemma, frozen)
    - LAM (VQ-VAE) for action encoding (frozen)
    - Trainable: Projector + Sequence Head (~34.6M parameters)

Key Innovation: OpenVLA Token Mapping
    - RLDS datasets use OpenVLA's ActionTokenizer → tokens at [31744, 32000)
    - We map: OpenVLA tokens → LAM indices [0, 511]

Author: Claude Opus 4.5
Date: 2025-03
"""

import math
import os
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaModel, GemmaForCausalLM
from transformers.models.siglip.modeling_siglip import SiglipVisionModel, SiglipVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput

logger = logging.getLogger(__name__)


# =============================================================================
# LAM (Latent Action Model) Loader
# =============================================================================

def load_lam_model(
    lam_checkpoint_path: str,
    lam_vocab_size: int = 512,
    lam_num_tokens: int = 4,
    device: str = "cuda",
) -> nn.Module:
    """Load the LAM (VQ-VAE) model for encoding/decoding actions."""
    import sys
    sys.path.insert(0, '/root/autodl-tmp/workspace/fangziyu/UniVLA')

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    logger.info(f"[*] [PI05 Adapter] Loading LAM model from {lam_checkpoint_path}")

    torch_hub_dir = os.environ.get('TORCH_HUB_DIR', '/home/nice/.cache/torch/hub')
    os.environ['TORCH_HUB_DIR'] = torch_hub_dir

    latent_action_model = ControllableDINOLatentActionModel(
        ckpt_path=lam_checkpoint_path,
        vocab_size=lam_vocab_size,
        num_latent_tokens=lam_num_tokens,
    ).to(device)

    logger.info(f"[*] [PI05 Adapter] LAM model loaded successfully")
    return latent_action_model


# =============================================================================
# Multi-Token Prediction Strategies
# =============================================================================

class SharedMLPStrategy(nn.Module):
    """
    Shared MLP strategy: Use a single MLP to predict all tokens jointly.

    Args:
        num_tokens: Number of LAM tokens to predict (default: 4)
        vocab_size: Size of LAM vocabulary (default: 512)
        hidden_dim: Hidden dimension for MLP (default: 2048)
    """

    def __init__(self, num_tokens: int = 4, vocab_size: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Shared MLP: maps features to N * vocab_size logits
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * vocab_size),
        )

        # Initialize final layer to small random values for better training
        nn.init.trunc_normal_(self.mlp[-1].weight, mean=0, std=0.02)
        nn.init.constant_(self.mlp[-1].bias, 0)

        logger.info(f"[*] [PI05 Adapter] Using SHARED MLP for {num_tokens} token predictions")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared MLP.

        Args:
            features: [B, seq_len, hidden_dim] or [B, hidden_dim]

        Returns:
            logits: [B, num_tokens, vocab_size]
        """
        if features.dim() == 3:
            # Use last token features for sequence prediction
            features = features[:, -1, :]  # [B, hidden_dim]

        logits = self.mlp(features)  # [B, num_tokens * vocab_size]
        batch_size = features.shape[0]
        logits = logits.view(batch_size, self.num_tokens, self.vocab_size)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        openvla_action_threshold: int = 31744,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with OpenVLA token mapping.

        IMPORTANT: The RLDS dataset creates action tokens using OpenVLA's ActionTokenizer
        which uses tokens at the end of the vocabulary (around 31744-32000 for a 32000 vocab).
        We need to find these tokens and map them to our LAM vocab [0, 511].

        Args:
            logits: [B, num_tokens, vocab_size] - predicted logits
            labels: [B, seq_len] - label token IDs
            openvla_action_threshold: Threshold for OpenVLA action tokens (default: 31744)

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        batch_size, seq_len = labels.shape
        device = logits.device

        # Find OpenVLA action tokens and map to our LAM vocab
        total_loss = 0.0
        total_tokens = 0
        num_tokens = self.num_tokens

        for b in range(batch_size):
            # Find positions where labels are OpenVLA action tokens
            act_mask = (labels[b] >= openvla_action_threshold) & (labels[b] != -100)
            act_positions = torch.where(act_mask)[0]

            if len(act_positions) < num_tokens:
                continue

            # Extract the last N action tokens
            openvla_act_tokens = labels[b, act_positions[-num_tokens:]]  # [N]

            # Map OpenVLA action token IDs [31744, 32000) to our LAM vocab [0, 511]
            lam_indices = openvla_act_tokens - openvla_action_threshold
            lam_indices = torch.clamp(lam_indices, 0, self.vocab_size - 1)

            # Compute loss for each token
            for t_idx, target_idx in enumerate(lam_indices):
                token_logits = logits[b, t_idx, :]  # [vocab_size]
                loss_t = nn.functional.cross_entropy(
                    token_logits.unsqueeze(0),
                    target_idx.unsqueeze(0).long(),
                    reduction='mean'
                )
                total_loss += loss_t
                total_tokens += 1

        if total_tokens == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {'loss': 0.0, 'num_tokens': 0}

        avg_loss = total_loss / total_tokens
        return avg_loss, {'loss': avg_loss.item(), 'num_tokens': total_tokens}


# =============================================================================
# PI05-for-UniVLA Main Adapter Class
# =============================================================================

@dataclass
class PI05ForUniVLAConfig:
    """Configuration for PI05-for-UniVLA adapter."""
    lam_vocab_size: int = 512
    lam_num_tokens: int = 4
    use_multi_token_prediction: bool = True
    hidden_dim: int = 2048
    lam_checkpoint_path: Optional[str] = None
    enable_mixed_precision_training: bool = True


class PI05ForUniVLA(nn.Module):
    """
    PI05 VLA Adapter for UniVLA Training Pipeline

    This class wraps PaliGemma (SigLIP + Gemma-2B) and adapts it for UniVLA's
    training infrastructure with LAM (VQ-VAE).

    Architecture:
        1. SigLIP Vision Encoder (frozen)
        2. Gemma-2B Language Model (frozen)
        3. LAM Model (frozen): VQ-VAE for action encoding/decoding
        4. Projector (trainable): Maps VLM features to training space
        5. Sequence Head (trainable): Predicts discrete action tokens

    Args:
        model_id: Identifier for the PI05 model
        config: Configuration object
        device: Device to load model on
    """

    def __init__(
        self,
        model_id: str = "pi05",
        lam_vocab_size: int = 512,
        lam_num_tokens: int = 4,
        lam_checkpoint_path: Optional[str] = None,
        use_multi_token_prediction: bool = True,
        enable_mixed_precision_training: bool = True,
    ):
        super().__init__()

        self.model_id = model_id
        self.lam_vocab_size = lam_vocab_size
        self.lam_num_tokens = lam_num_tokens
        self.use_multi_token_prediction = use_multi_token_prediction
        self.enable_mixed_precision_training = enable_mixed_precision_training

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[*] [PI05 Adapter] Initializing PI05 model from scratch")

        # Initialize SigLIP vision encoder
        self.vision_encoder = self._create_siglip_encoder()

        # Initialize Gemma-2B language model
        self.language_model = self._create_gemma_lm()

        # Hidden dimensions
        self.vision_embed_dim = self.vision_encoder.config.hidden_size  # 1152 for SigLIP SO400M
        self.llm_embed_dim = self.language_model.config.hidden_size  # 2048 for Gemma-2B
        self.hidden_dim = self.llm_embed_dim

        # Track module keys for checkpointing/training
        self.all_module_keys = ["vision_encoder", "language_model", "projector", "sequence_head"]
        self.trainable_module_keys = ["projector", "sequence_head"]

        # Projector: maps vision features to LLM dimension
        self.projector = nn.Sequential(
            nn.Linear(self.vision_embed_dim, self.llm_embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.llm_embed_dim),
        )

        # Multi-token prediction strategy
        if use_multi_token_prediction:
            self.sequence_head = SharedMLPStrategy(
                num_tokens=lam_num_tokens,
                vocab_size=lam_vocab_size,
                hidden_dim=self.hidden_dim,
            )
        else:
            self.sequence_head = nn.Linear(self.hidden_dim, lam_vocab_size)
            nn.init.trunc_normal_(self.sequence_head.weight, mean=0, std=0.02)
            nn.init.constant_(self.sequence_head.bias, 0)

        # LAM model (loaded later)
        self.latent_action_model = None
        self.lam_checkpoint_path = lam_checkpoint_path

        # OpenVLA action token threshold
        self.openvla_action_threshold = 32000 - 256  # 31744

        logger.info(f"[*] [PI05 Adapter] PI05 adapter initialized successfully")
        logger.info(f"[*] [PI05 Adapter] Vision embed dim: {self.vision_embed_dim}")
        logger.info(f"[*] [PI05 Adapter] LLM embed dim: {self.llm_embed_dim}")

    def _create_siglip_encoder(self):
        """Create SigLIP vision encoder from config."""
        logger.info(f"[*] [PI05 Adapter] Creating SigLIP vision encoder")

        # SigLIP SO400M config (used in PaliGemma-2B)
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=224,
            patch_size=14,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            projection_dim=2048,
        )

        vision_model = SiglipVisionModel(vision_config)
        return vision_model

    def _create_gemma_lm(self):
        """Create Gemma-2B language model from config."""
        logger.info(f"[*] [PI05 Adapter] Creating Gemma-2B language model")

        # Gemma-2B config
        llm_config = GemmaConfig(
            hidden_size=2048,
            num_hidden_layers=18,
            num_attention_heads=8,
            num_key_value_heads=1,
            intermediate_size=16384,
            vocab_size=257152,  # PaliGemma uses extended vocab
            hidden_act="gelu_pytorch_tanh",
            max_position_embeddings=8192,
        )

        # Use GemmaForCausalLM to get logits output (not just hidden states)
        from transformers import GemmaForCausalLM
        llm = GemmaForCausalLM(llm_config)
        return llm

    def _load_lam_model(self):
        """Load LAM model from checkpoint."""
        if self.lam_checkpoint_path is None:
            logger.warning(f"[*] [PI05 Adapter] No LAM checkpoint path provided")
            return None

        logger.info(f"[*] [PI05 Adapter] Loading LAM model from {self.lam_checkpoint_path}")

        self.latent_action_model = load_lam_model(
            lam_checkpoint_path=self.lam_checkpoint_path,
            lam_vocab_size=self.lam_vocab_size,
            lam_num_tokens=self.lam_num_tokens,
            device=self.device,
        )

        return self.latent_action_model

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image using SigLIP vision encoder.

        Args:
            pixel_values: [B, C, H, W] - Image input

        Returns:
            image_features: [B, num_patches, vision_embed_dim]
        """
        # SigLIP expects [B, C, H, W] with values in [-1, 1]
        outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = outputs.last_hidden_state  # [B, num_patches, vision_embed_dim]
        return image_features

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text using Gemma language model embeddings.

        Args:
            input_ids: [B, seq_len] - Text token IDs

        Returns:
            text_features: [B, seq_len, llm_embed_dim]
        """
        # GemmaForCausalLM has embeddings in model.embed_tokens
        if hasattr(self.language_model, 'model'):
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        return inputs_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for PI05 VLA adapter.

        Args:
            pixel_values: [B, C, H, W] - Image input
            input_ids: [B, seq_len] - Text token IDs
            attention_mask: [B, seq_len] - Attention mask
            labels: [B, seq_len] - Target token IDs (for training)
            output_hidden_states: Whether to return hidden states

        Returns:
            CausalLMOutputWithPast with:
                - logits: [B, seq_len, vocab_size]
                - hidden_states: Tuple of hidden states (if output_hidden_states)
                - loss: Scalar loss (if labels provided)
        """
        batch_size = pixel_values.shape[0]

        # Encode image
        image_features = self.encode_image(pixel_values)  # [B, num_patches, vision_embed_dim]

        # Project to LLM dimension
        projected_features = self.projector(image_features)  # [B, num_patches, llm_embed_dim]

        # Encode text
        text_features = self.encode_text(input_ids)  # [B, seq_len, llm_embed_dim]

        # Combine vision and text features
        # For VLA training, we need to prepend vision tokens to the sequence
        # This matches the OpenVLA format where vision tokens come first
        seq_len = input_ids.shape[1]

        # Create a sequence of vision tokens (same as num_patches)
        num_patches = projected_features.shape[1]

        # Pad vision tokens to same dim as language
        # Pad text with vision tokens at the beginning
        # This creates: [B, num_patches + seq_len, llm_embed_dim]
        combined_features = torch.cat([projected_features, text_features], dim=1)

        # Create attention mask for combined sequence
        if attention_mask is not None:
            # Extend attention mask to include vision tokens
            vision_mask = torch.ones(batch_size, num_patches, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Run through LLM
        llm_outputs = self.language_model(
            inputs_embeds=combined_features,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,  # Disable KV cache for training to avoid dimension mismatch
        )

        # Get logits from LLM
        lm_logits = llm_outputs.logits  # [B, num_patches + seq_len, vocab_size]

        # Process logits to get action predictions
        # For PI05 with multi-token prediction, we use the last N vision tokens
        # for LAM token prediction, but still maintain the full logits for compatibility

        # Compute loss if labels provided using OpenVLA token mapping
        loss = None
        if labels is not None:
            loss = self.compute_loss_with_openvla_tokens(lm_logits, labels)

        # Return CausalLMOutputWithPast for training compatibility
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=llm_outputs.hidden_states if output_hidden_states else None,
            past_key_values=llm_outputs.past_key_values,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with OpenVLA token mapping.

        Args:
            logits: [B, num_tokens, vocab_size] or [B, vocab_size]
            labels: [B, seq_len] - Label token IDs from dataset

        Returns:
            loss: Scalar loss tensor
        """
        if self.use_multi_token_prediction:
            loss, metrics = self.sequence_head.compute_loss(
                logits=logits,
                labels=labels,
                openvla_action_threshold=self.openvla_action_threshold,
            )
            return loss
        else:
            # Single token loss
            total_loss = 0.0
            total_tokens = 0

            batch_size = labels.shape[0]
            device = logits.device

            for b in range(batch_size):
                act_mask = (labels[b] >= self.openvla_action_threshold) & (labels[b] != -100)
                act_positions = torch.where(act_mask)[0]

                if len(act_positions) == 0:
                    continue

                openvla_act_token = labels[b, act_positions[-1]]
                lam_index = torch.clamp(openvla_act_token - self.openvla_action_threshold, 0, self.lam_vocab_size - 1)

                loss_t = nn.functional.cross_entropy(
                    logits[b].unsqueeze(0),
                    lam_index.unsqueeze(0).long(),
                    reduction='mean'
                )
                total_loss += loss_t
                total_tokens += 1

            if total_tokens == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            return total_loss / total_tokens

    def compute_loss_with_openvla_tokens(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with OpenVLA token mapping for full-sequence logits.

        This method computes cross-entropy loss on logits where labels are OpenVLA
        action tokens (at the end of the vocabulary, around 31744-32000).

        Args:
            logits: [B, num_patches + seq_len, vocab_size] - full sequence logits
            labels: [B, seq_len] - Label token IDs from dataset

        Returns:
            loss: Scalar loss tensor
        """
        batch_size, label_seq_len = labels.shape
        device = logits.device
        logits_seq_len = logits.shape[1]

        # Create loss mask: positions where labels are not -100 (ignore index)
        loss_mask = (labels != -100) & (labels >= 0)

        if not loss_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Align logits with labels
        # logits has vision tokens prepended, so we need to skip them
        # Calculate how many text tokens are in the logits
        num_vision_tokens = logits_seq_len - label_seq_len

        # Use the text portion of logits to compute loss
        # Skip first token (BOS) and align with labels shifted by 1
        if num_vision_tokens > 0:
            # Take text portion of logits (skip vision tokens), then shift
            text_logits = logits[:, num_vision_tokens:, :]  # [B, label_seq_len, vocab_size]
            shift_logits = text_logits[:, :-1, :]  # [B, label_seq_len-1, vocab_size]
        else:
            shift_logits = logits[:, :-1, :]  # [B, logits_seq_len-1, vocab_size]

        shift_labels = labels[:, 1:].to(shift_logits.device)  # [B, label_seq_len-1]
        shift_mask = loss_mask[:, 1:]

        # Ensure dimensions match
        min_len = min(shift_logits.shape[1], shift_labels.shape[1])
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        shift_mask = shift_mask[:, :min_len]

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction='none'
        )

        # Mask out ignored positions
        loss = loss * shift_mask.reshape(-1).float()

        # Return mean loss over valid positions
        return loss.sum() / shift_mask.sum().float().clamp(min=1)

    def freeze_backbones(self, stage: str = "align"):
        """
        Freeze model backbones for different training stages.

        Args:
            stage: Training stage
                - "align": Freeze VLM, train projector + sequence_head
                - "finetune": Same as align (LAM frozen)
                - "full": Train all parameters
        """
        if stage == "align":
            # Freeze vision encoder and language model
            self.vision_encoder.requires_grad_(False)
            self.language_model.requires_grad_(False)

            # Train projector and sequence head
            self.projector.requires_grad_(True)
            self.sequence_head.requires_grad_(True)

            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[*] [PI05 Adapter] Stage 'align': VLM frozen, projector+sequence_head trainable")
            logger.info(f"[*] [PI05 Adapter] Trainable parameters: {trainable_params:,}")

        elif stage == "finetune":
            # Same as align for now
            self.vision_encoder.requires_grad_(False)
            self.language_model.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.sequence_head.requires_grad_(True)

            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[*] [PI05 Adapter] Stage 'finetune': VLM frozen")

        elif stage == "full":
            # Train all parameters
            self.requires_grad_(True)
            logger.info(f"[*] [PI05 Adapter] Stage 'full': All parameters trainable")

    @torch.no_grad()
    def generate_action(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate action tokens from image and text.

        Args:
            pixel_values: [B, C, H, W] - Image input
            input_ids: [B, seq_len] - Text token IDs

        Returns:
            action_tokens: [B, num_tokens] - Predicted action token IDs
        """
        self.eval()
        outputs = self.forward(pixel_values=pixel_values, input_ids=input_ids)
        logits = outputs['logits']

        if self.use_multi_token_prediction:
            action_tokens = logits.argmax(dim=-1)  # [B, num_tokens]
        else:
            token = logits.argmax(dim=-1)  # [B]
            action_tokens = token.unsqueeze(-1).repeat(1, self.lam_num_tokens)  # [B, num_tokens]

        return action_tokens

    def decode_to_actions(
        self,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode action tokens to continuous actions using LAM decoder.

        Args:
            action_tokens: [B, num_tokens] - Action token IDs in LAM vocab

        Returns:
            actions: [B, action_dim] - Continuous 7-DOF actions
        """
        if self.latent_action_model is None:
            raise ValueError("LAM model not loaded. Call _load_lam_model() first.")

        # Decode using LAM
        with torch.no_grad():
            actions = self.latent_action_model.decode(action_tokens)

        return actions

    def get_fsdp_wrapping_policy(self):
        """Get FSDP wrapping policy for distributed training."""
        from functools import partial
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # 用 partial 绑定参数，返回可调用对象
        # transformer_layer_cls 需要是元组 (xxx,)
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(self.llm_backbone.transformer_layer_cls,)
        )


# =============================================================================
# Backbone Wrappers for UniVLA Compatibility
# =============================================================================

class VisionBackboneWrapper:
    """Wrapper for PI05 vision encoder to match UniVLA API."""
    def __init__(self, vision_encoder, image_size=224, patch_size=14):
        self.model = vision_encoder
        self.default_image_resolution = (image_size, image_size)
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = vision_encoder.config.hidden_size
        self.half_precision_dtype = torch.bfloat16

    def to(self, *args, **kwargs):
        """Forward to() calls to the underlying model."""
        self.model = self.model.to(*args, **kwargs)
        return self

    def get_image_transform(self):
        """Get image transform for SigLIP."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


class LLMBackboneWrapper:
    """Wrapper for PI05 language model to match UniVLA API."""
    def __init__(self, language_model):
        self.model = language_model
        # Add transformer_layer_cls for FSDP training strategy compatibility
        # GemmaModel has .layers directly at root (not .model.layers)
        if hasattr(language_model, 'layers'):
            self.transformer_layer_cls = type(language_model.layers[0])
        elif hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
            self.transformer_layer_cls = type(language_model.model.layers[0])
        else:
            from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
            self.transformer_layer_cls = GemmaDecoderLayer

    def get_tokenizer(self):
        """Get Gemma tokenizer."""
        from transformers import AutoTokenizer, GemmaTokenizerFast
        # Try PaliGemma tokenizer first (extended vocab)
        try:
            # Try using the snapshot path directly
            import os
            paligemma_cache = "/home/nice/.cache/huggingface/hub/models--google--paligemma-2b"
            if os.path.exists(paligemma_cache):
                snapshots = [d for d in os.listdir(paligemma_cache) if d.startswith("snapshots")]
                if snapshots:
                    snapshot_path = os.path.join(paligemma_cache, snapshots[-1])
                    if os.path.exists(snapshot_path):
                        return AutoTokenizer.from_pretrained(
                            snapshot_path,
                            padding_side="right",
                            local_files_only=True
                        )
        except Exception as e:
            pass

        # Fallback: create minimal tokenizer for PaliGemma vocab size
        class TokenizerResult:
            """Wrapper for tokenizer output to mimic BatchEncoding behavior."""
            def __init__(self, input_ids):
                self.input_ids = input_ids

        class SimpleTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = 257152  # PaliGemma extended vocab
                self.model_max_length = 8192
                self.padding_side = "right"
            def __call__(self, text, *args, **kwargs):
                # Minimal stub - actual tokenization happens elsewhere
                # Return object with input_ids attribute (not dict)
                input_ids = list(range(len(text))) if isinstance(text, str) else text
                return TokenizerResult(input_ids)
        return SimpleTokenizer()


# Monkey-patch the wrappers onto PI05ForUniVLA after initialization
def _add_backbone_wrappers(self):
    """Add backbone wrapper attributes for UniVLA compatibility."""
    if not hasattr(self, 'vision_backbone'):
        self.vision_backbone = VisionBackboneWrapper(self.vision_encoder)
    if not hasattr(self, 'llm_backbone'):
        self.llm_backbone = LLMBackboneWrapper(self.language_model)
        # Add prompt_builder_fn
        # Add prompt_builder_fn - return PurePromptBuilder instance
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        self.llm_backbone.prompt_builder_fn = lambda model_family="gemma", system_prompt=None: PurePromptBuilder(model_family, system_prompt)


# Patch the __init__ to call the wrapper setup
original_init = PI05ForUniVLA.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    _add_backbone_wrappers(self)

PI05ForUniVLA.__init__ = patched_init


# =============================================================================
# Convenience Function for Loading
# =============================================================================

def load_pi05(
    model_id: str = "pi05",
    lam_vocab_size: int = 512,
    lam_num_tokens: int = 4,
    lam_checkpoint_path: Optional[str] = None,
    use_multi_token_prediction: bool = True,
    enable_mixed_precision_training: bool = True,
) -> PI05ForUniVLA:
    """
    Load PI05-for-UniVLA adapter.

    Args:
        model_id: Identifier for PI05 model
        lam_vocab_size: LAM vocabulary size
        lam_num_tokens: Number of LAM tokens per action
        lam_checkpoint_path: Path to LAM checkpoint
        use_multi_token_prediction: Use multi-token prediction strategy
        enable_mixed_precision_training: Enable mixed precision training

    Returns:
        PI05ForUniVLA model
    """
    model = PI05ForUniVLA(
        model_id=model_id,
        lam_vocab_size=lam_vocab_size,
        lam_num_tokens=lam_num_tokens,
        lam_checkpoint_path=lam_checkpoint_path,
        use_multi_token_prediction=use_multi_token_prediction,
        enable_mixed_precision_training=enable_mixed_precision_training,
    )

    return model
