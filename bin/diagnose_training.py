#!/usr/bin/env python3
"""
Diagnostic script to identify training issues.

Checks:
- Data loading and preprocessing
- Model initialization
- Forward pass
- Loss computation
- Gradient flow
- Learning rate schedule
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import BertConfig

from foldingdiff.enhanced_datasets import create_enhanced_dataset
from foldingdiff.enhanced_models import BertForFlowMatchingEnhancedTraining


def check_data(dataset, n_samples=5):
    """Check dataset for issues."""
    print("\n" + "="*80)
    print("CHECKING DATA")
    print("="*80)
    
    for i in range(min(n_samples, len(dataset))):
        batch = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Angles shape: {batch['angles'].shape}")
        print(f"  Angles range: [{batch['angles'].min():.3f}, {batch['angles'].max():.3f}]")
        print(f"  Angles mean: {batch['angles'].mean():.3f}, std: {batch['angles'].std():.3f}")
        
        # Check for NaN/Inf
        if torch.isnan(batch['angles']).any():
            print(f"  ⚠️  WARNING: NaN values in angles!")
        if torch.isinf(batch['angles']).any():
            print(f"  ⚠️  WARNING: Inf values in angles!")
        
        # Check coords if present
        if 'coords_computed' in batch:
            coords = batch['coords_computed']
            print(f"  Coords shape: {coords.shape}")
            print(f"  Coords range: [{coords.min():.3f}, {coords.max():.3f}]")
            if torch.isnan(coords).any():
                print(f"  ⚠️  WARNING: NaN values in coords!")
        
        # Check motif mask if present
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            print(f"  Motif ratio: {motif_ratio:.2%}")
    
    print("\n✓ Data check complete")


def check_model_init(model):
    """Check model initialization."""
    print("\n" + "="*80)
    print("CHECKING MODEL INITIALIZATION")
    print("="*80)
    
    param_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            param_stats.append((name, mean, std))
            
            # Check for bad initialization
            if abs(mean) > 1.0:
                print(f"  ⚠️  Large mean in {name}: {mean:.3f}")
            if std > 10.0:
                print(f"  ⚠️  Large std in {name}: {std:.3f}")
            if std < 1e-6:
                print(f"  ⚠️  Very small std in {name}: {std:.3e}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Show statistics for key layers
    print("\nKey layer statistics:")
    for name, mean, std in param_stats[:10]:
        print(f"  {name:50s} mean={mean:7.3f}, std={std:7.3f}")
    
    print("\n✓ Model initialization check complete")


def check_forward_pass(model, batch, device='cpu'):
    """Check forward pass."""
    print("\n" + "="*80)
    print("CHECKING FORWARD PASS")
    print("="*80)
    
    model = model.to(device)
    model.eval()
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Add batch dimension if needed
    if batch['angles'].ndim == 2:
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 else v 
                 for k, v in batch.items()}
    
    x_0 = batch['angles']
    batch_size = x_0.shape[0]
    
    # Sample time and noise
    t = torch.rand(batch_size, device=device)
    x_1 = torch.randn_like(x_0)
    
    # Get interpolant
    from foldingdiff.flow_matching import FlowMatchingSchedule
    flow_schedule = FlowMatchingSchedule()
    x_t = flow_schedule.get_interpolant(x_0, x_1, t)
    
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Input x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model.forward(
                x_t, t,
                attention_mask=batch['attn_mask'],
                coords=batch.get('coords_computed', None),
                aa_types=batch.get('aa_types', None),
                secondary_structure=batch.get('secondary_structure', None),
                motif_mask=batch.get('motif_mask', None),
                motif_features=batch.get('motif_angles', None),
            )
            
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output mean: {output.mean():.3f}, std: {output.std():.3f}")
            
            if torch.isnan(output).any():
                print("  ⚠️  WARNING: NaN in output!")
            if torch.isinf(output).any():
                print("  ⚠️  WARNING: Inf in output!")
            
            print("\n✓ Forward pass successful")
            return True
            
        except Exception as e:
            print(f"  ❌ ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_loss_computation(model, batch, device='cpu'):
    """Check loss computation and gradients."""
    print("\n" + "="*80)
    print("CHECKING LOSS AND GRADIENTS")
    print("="*80)
    
    model = model.to(device)
    model.train()
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Add batch dimension if needed
    if batch['angles'].ndim == 2:
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 else v 
                 for k, v in batch.items()}
    
    # Compute loss
    try:
        loss = model.training_step(batch, 0)
        
        if loss is None:
            print("  ⚠️  WARNING: Loss is None (batch skipped)")
            return False
        
        print(f"Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("  ❌ ERROR: Loss is NaN!")
            return False
        if torch.isinf(loss):
            print("  ❌ ERROR: Loss is Inf!")
            return False
        
        # Compute gradients
        loss.backward()
        
        # Check gradients
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.abs().max().item()
                grad_stats.append((name, grad_mean, grad_std, grad_max))
                
                if torch.isnan(param.grad).any():
                    print(f"  ⚠️  NaN gradient in {name}")
                if grad_max > 100:
                    print(f"  ⚠️  Large gradient in {name}: {grad_max:.3f}")
        
        # Show gradient statistics
        print("\nGradient statistics (first 10 layers):")
        for name, mean, std, max_val in grad_stats[:10]:
            print(f"  {name:50s} mean={mean:8.3e}, std={std:8.3e}, max={max_val:8.3e}")
        
        # Check if gradients are flowing
        has_grad = sum(1 for _, _, _, m in grad_stats if m > 1e-10)
        print(f"\nLayers with non-zero gradients: {has_grad}/{len(grad_stats)}")
        
        if has_grad < len(grad_stats) * 0.5:
            print("  ⚠️  WARNING: Many layers have zero gradients (vanishing gradient problem)")
        
        print("\n✓ Loss and gradient check complete")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_learning_rate(model, steps=10):
    """Check learning rate schedule."""
    print("\n" + "="*80)
    print("CHECKING LEARNING RATE SCHEDULE")
    print("="*80)
    
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config['optimizer']
    
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.3e}")
    
    if 'lr_scheduler' in optimizer_config:
        scheduler = optimizer_config['lr_scheduler']['scheduler']
        print(f"Scheduler: {type(scheduler).__name__}")
        
        # Simulate a few steps
        lrs = []
        for step in range(steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        print(f"\nLR progression over {steps} steps:")
        for i, lr in enumerate(lrs):
            print(f"  Step {i}: {lr:.3e}")
    
    print("\n✓ Learning rate check complete")


def main():
    print("="*80)
    print("TRAINING DIAGNOSTICS")
    print("="*80)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_enhanced_dataset(
        split=None,
        pad=128,
        min_length=40,
        toy=10,  # Small toy dataset for testing
        compute_coords=True,
        compute_ss=False,
        use_motif_scaffolding=True,
        motif_length_range=(5, 20),
        motif_prob=0.8,
        timesteps=1000,
        beta_schedule="cosine",
    )
    
    # Check data
    check_data(dataset)
    
    # Create model
    print("\nCreating model...")
    config = BertConfig(
        hidden_size=384,
        num_hidden_layers=6,  # Smaller for testing
        num_attention_heads=12,
        intermediate_size=384 * 4,
        max_position_embeddings=128,
    )
    
    embedding_config = {
        'use_sequence': False,
        'use_coords': True,
        'use_local_frames': True,
        'use_pairwise': True,
        'use_secondary_structure': False,
    }
    
    model = BertForFlowMatchingEnhancedTraining(
        config=config,
        ft_is_angular=[True, True, True, False, False, False],
        ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
        use_enhanced_embedding=True,
        embedding_config=embedding_config,
        use_motif_conditioning=True,
        lr=1e-4,
        epochs=10,
        steps_per_epoch=len(dataset),
        lr_scheduler="LinearWarmup",
        guidance_dropout=0.1,
    )
    
    # Check model initialization
    check_model_init(model)
    
    # Get a batch
    batch = dataset[0]
    
    # Check forward pass
    forward_ok = check_forward_pass(model, batch, device)
    
    if forward_ok:
        # Check loss and gradients
        check_loss_computation(model, batch, device)
    
    # Check learning rate
    check_learning_rate(model)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print("\nRecommendations:")
    print("1. If loss is NaN/Inf: Check data normalization and model initialization")
    print("2. If gradients are zero: Check for vanishing gradients, try higher LR")
    print("3. If loss not decreasing: Try higher LR (1e-4 to 5e-4), check data quality")
    print("4. Monitor training with: tensorboard --logdir results/enhanced_flow/*/logs")


if __name__ == "__main__":
    main()
