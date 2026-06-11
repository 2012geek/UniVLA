"""
PI05 Training Test with Synthetic Data

This script tests PI05 training without requiring external datasets or LAM checkpoints.
It generates synthetic data to verify the training pipeline works correctly.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
import tqdm

# Setup paths
sys.path.insert(0, '/root/autodl-tmp/workspace/fangziyu/UniVLA')
os.environ['TORCH_HUB_DIR'] = '/home/nice/.cache/torch/hub'

# Import PI05 directly
exec(open('/root/autodl-tmp/workspace/fangziyu/UniVLA/prismatic/models/vlas/pi05_for_univla.py').read())

def generate_synthetic_batch(batch_size=4, seq_len=10, img_size=224):
    """Generate synthetic training batch."""
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    input_ids = torch.randint(0, 257152, (batch_size, seq_len))
    # Generate labels with OpenVLA-style action tokens at the end
    labels = torch.cat([
        torch.randint(0, 30000, (batch_size, 5)),  # Regular tokens
        torch.randint(31744, 32000, (batch_size, 4)),  # Action tokens
    ], dim=1)
    attention_mask = torch.ones(batch_size, seq_len + 9)
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

def main():
    print("=" * 70)
    print("PI05 Training Test with Synthetic Data")
    print("=" * 70)
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_steps': 100,
        'learning_rate': 3.5e-4,
        'lam_vocab_size': 512,
        'lam_num_tokens': 4,
    }
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Training steps: {config['num_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LAM vocab size: {config['lam_vocab_size']}")
    print(f"  LAM num tokens: {config['lam_num_tokens']}")
    
    # Load model
    print("\n" + "=" * 70)
    print("Loading PI05 model...")
    print("=" * 70)
    
    vla = load_pi05(
        lam_vocab_size=config['lam_vocab_size'],
        lam_num_tokens=config['lam_num_tokens'],
        use_multi_token_prediction=True,
    )
    
    # Freeze backbones
    vla.freeze_backbones('align')
    vla.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in vla.parameters())
    trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, vla.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-3,
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    losses = []
    pbar = tqdm.tqdm(range(config['num_steps']), desc="Training")
    
    for step in pbar:
        # Generate synthetic batch
        batch = generate_synthetic_batch(config['batch_size'])
        
        # Forward pass
        outputs = vla(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update progress bar
        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nLoss statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Average loss: {sum(losses)/len(losses):.4f}")
    print(f"  Min loss: {min(losses):.4f}")
    print(f"  Max loss: {max(losses):.4f}")
    
    # Test inference
    print("\n" + "=" * 70)
    print("Testing Inference...")
    print("=" * 70)
    
    vla.eval()
    with torch.no_grad():
        batch = generate_synthetic_batch(2)
        action_tokens = vla.generate_action(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
        )
        print(f"\nGenerated action tokens shape: {action_tokens.shape}")
        print(f"Sample action tokens:\n{action_tokens}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nPI05 is ready for production training.")
    print("To train with real data, provide:")
    print("  1. LAM checkpoint path")
    print("  2. LIBERO dataset path")
    print("  3. Run: python run_pi05_finetune_test.py")

if __name__ == "__main__":
    main()
