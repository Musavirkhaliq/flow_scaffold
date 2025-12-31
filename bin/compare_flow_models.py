#!/usr/bin/env python3
"""
Compare different flow matching models to evaluate improvements.

Compares:
1. Original enhanced model
2. Advanced model with 2024-2025 improvements
3. Baseline model (if available)
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
from typing import Dict, List, Any

import torch
import numpy as np
from transformers import BertConfig

from foldingdiff.enhanced_models import BertForFlowMatchingEnhancedTraining
from foldingdiff.enhanced_models_v2 import BertForAdvancedFlowMatchingTraining
from foldingdiff.enhanced_datasets import create_enhanced_dataset


def create_model(model_type: str, config: BertConfig, **kwargs) -> torch.nn.Module:
    """Create model based on type"""
    
    if model_type == "enhanced":
        return BertForFlowMatchingEnhancedTraining(
            config=config,
            ft_is_angular=[True, True, True, False, False, False],
            ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            use_enhanced_embedding=True,
            embedding_config={
                'use_sequence': False,
                'use_coords': True,
                'use_local_frames': True,
                'use_pairwise': True,
                'use_secondary_structure': False,
            },
            use_motif_conditioning=True,
            **kwargs
        )
    
    elif model_type == "advanced":
        return BertForAdvancedFlowMatchingTraining(
            config=config,
            ft_is_angular=[True, True, True, False, False, False],
            ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            use_enhanced_embedding=True,
            embedding_config={
                'use_sequence': True,
                'use_coords': True,
                'use_local_frames': True,
                'use_pairwise': True,
                'use_secondary_structure': False,
            },
            use_motif_conditioning=True,
            
            # Advanced features
            use_sequence_augmentation=True,
            use_geometric_inverse_design=True,
            use_multiscale_attention=True,
            plm_model="facebook/esm2_t12_35M_UR50D",
            fusion_mode="cross_attn",
            
            # Advanced training
            use_multiscale_loss=True,
            use_consistency_loss=True,
            use_geometric_loss=True,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def benchmark_model(
    model: torch.nn.Module,
    test_batch: Dict[str, torch.Tensor],
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark model performance"""
    
    model.eval()
    device = next(model.parameters()).device
    
    # Move batch to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in test_batch.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.forward(
                batch['angles'],
                batch['timestep'],
                batch['attn_mask'],
                coords=batch.get('coords_computed'),
                aa_types=batch.get('aa_types'),
                sequences=batch.get('sequences'),
                secondary_structure=batch.get('secondary_structure'),
                motif_mask=batch.get('motif_mask'),
                motif_features=batch.get('motif_angles'),
                motif_coords=batch.get('motif_coords'),
            )
    
    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model.forward(
                batch['angles'],
                batch['timestep'],
                batch['attn_mask'],
                coords=batch.get('coords_computed'),
                aa_types=batch.get('aa_types'),
                sequences=batch.get('sequences'),
                secondary_structure=batch.get('secondary_structure'),
                motif_mask=batch.get('motif_mask'),
                motif_features=batch.get('motif_angles'),
                motif_coords=batch.get('motif_coords'),
            )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_used = 0.0
    
    # Model size
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    return {
        'forward_time_ms': avg_time * 1000,
        'memory_gb': memory_used,
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'output_shape': list(output.shape),
    }


def evaluate_sample_quality(
    model: torch.nn.Module,
    test_batch: Dict[str, torch.Tensor],
    num_samples: int = 5
) -> Dict[str, float]:
    """Evaluate sample quality metrics"""
    
    model.eval()
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in test_batch.items()}
    
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample different timesteps
            t = torch.rand(batch['angles'].shape[0], device=device)
            
            output = model.forward(
                batch['angles'],
                t,
                batch['attn_mask'],
                coords=batch.get('coords_computed'),
                aa_types=batch.get('aa_types'),
                sequences=batch.get('sequences'),
                secondary_structure=batch.get('secondary_structure'),
                motif_mask=batch.get('motif_mask'),
                motif_features=batch.get('motif_angles'),
                motif_coords=batch.get('motif_coords'),
            )
            samples.append(output.cpu())
    
    samples = torch.stack(samples)  # [num_samples, batch, seq_len, features]
    
    # Compute diversity (standard deviation across samples)
    diversity = torch.std(samples, dim=0).mean().item()
    
    # Compute consistency (how similar predictions are for similar inputs)
    pairwise_diffs = []
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            diff = torch.mean((samples[i] - samples[j])**2).item()
            pairwise_diffs.append(diff)
    
    consistency = np.mean(pairwise_diffs)
    
    # Compute output statistics
    output_mean = torch.mean(samples).item()
    output_std = torch.std(samples).item()
    output_range = (torch.max(samples) - torch.min(samples)).item()
    
    return {
        'diversity': diversity,
        'consistency': consistency,
        'output_mean': output_mean,
        'output_std': output_std,
        'output_range': output_range,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare flow matching models")
    
    parser.add_argument("--models", nargs="+", default=["enhanced", "advanced"],
                       choices=["enhanced", "advanced"],
                       help="Models to compare")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="model_comparison.json")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create test configuration
    config = BertConfig(
        hidden_size=384,
        num_hidden_layers=6,  # Smaller for comparison
        num_attention_heads=12,
        intermediate_size=384 * 4,
        max_position_embeddings=128,
    )
    
    # Create test batch
    logger.info("Creating test batch...")
    
    test_batch = {
        'angles': torch.randn(args.batch_size, args.seq_len, 6),
        'timestep': torch.rand(args.batch_size),
        'attn_mask': torch.ones(args.batch_size, args.seq_len),
        'coords_computed': torch.randn(args.batch_size, args.seq_len, 4, 3),
        'aa_types': torch.randint(0, 20, (args.batch_size, args.seq_len)),
        'sequences': [
            ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), args.seq_len))
            for _ in range(args.batch_size)
        ],
        'motif_mask': torch.zeros(args.batch_size, args.seq_len, 1),
        'motif_angles': torch.randn(args.batch_size, args.seq_len, 6),
        'motif_coords': torch.randn(args.batch_size, 10, 4, 3),  # Smaller motif
    }
    
    # Add some motif positions
    test_batch['motif_mask'][:, 10:20, :] = 1.0
    
    results = {}
    
    for model_type in args.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {model_type.upper()} model")
        logger.info(f"{'='*50}")
        
        try:
            # Create model
            logger.info("Creating model...")
            model = create_model(
                model_type, config,
                lr=1e-4,
                epochs=1,
                steps_per_epoch=100
            )
            model.to(device)
            
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Benchmark performance
            logger.info("Benchmarking performance...")
            perf_metrics = benchmark_model(model, test_batch, args.num_runs)
            
            # Evaluate sample quality
            logger.info("Evaluating sample quality...")
            quality_metrics = evaluate_sample_quality(model, test_batch, args.num_samples)
            
            # Combine results
            results[model_type] = {
                'performance': perf_metrics,
                'quality': quality_metrics,
                'model_info': {
                    'type': model_type,
                    'config': {
                        'hidden_size': config.hidden_size,
                        'num_layers': config.num_hidden_layers,
                        'num_heads': config.num_attention_heads,
                    }
                }
            }
            
            # Print summary
            logger.info(f"\n{model_type.upper()} Results:")
            logger.info(f"  Forward time: {perf_metrics['forward_time_ms']:.2f} ms")
            logger.info(f"  Memory usage: {perf_metrics['memory_gb']:.2f} GB")
            logger.info(f"  Parameters: {perf_metrics['num_parameters']:,}")
            logger.info(f"  Model size: {perf_metrics['model_size_mb']:.1f} MB")
            logger.info(f"  Sample diversity: {quality_metrics['diversity']:.4f}")
            logger.info(f"  Sample consistency: {quality_metrics['consistency']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Save results
    logger.info(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    if len(results) > 1 and all('error' not in r for r in results.values()):
        logger.info(f"\n{'='*50}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*50}")
        
        models = list(results.keys())
        
        # Performance comparison
        logger.info("\nPerformance Metrics:")
        logger.info(f"{'Metric':<20} {'Enhanced':<15} {'Advanced':<15} {'Ratio':<10}")
        logger.info("-" * 65)
        
        if 'enhanced' in results and 'advanced' in results:
            enh = results['enhanced']['performance']
            adv = results['advanced']['performance']
            
            metrics = [
                ('Forward Time (ms)', 'forward_time_ms'),
                ('Memory (GB)', 'memory_gb'),
                ('Parameters', 'num_parameters'),
                ('Model Size (MB)', 'model_size_mb'),
            ]
            
            for name, key in metrics:
                enh_val = enh[key]
                adv_val = adv[key]
                ratio = adv_val / enh_val if enh_val > 0 else float('inf')
                
                if key == 'num_parameters':
                    logger.info(f"{name:<20} {enh_val:<15,} {adv_val:<15,} {ratio:<10.2f}x")
                else:
                    logger.info(f"{name:<20} {enh_val:<15.2f} {adv_val:<15.2f} {ratio:<10.2f}x")
        
        # Quality comparison
        logger.info("\nQuality Metrics:")
        logger.info(f"{'Metric':<20} {'Enhanced':<15} {'Advanced':<15} {'Improvement':<12}")
        logger.info("-" * 67)
        
        if 'enhanced' in results and 'advanced' in results:
            enh_q = results['enhanced']['quality']
            adv_q = results['advanced']['quality']
            
            quality_metrics = [
                ('Diversity', 'diversity'),
                ('Consistency', 'consistency'),
                ('Output Std', 'output_std'),
            ]
            
            for name, key in quality_metrics:
                enh_val = enh_q[key]
                adv_val = adv_q[key]
                
                if key == 'diversity':
                    improvement = "Better" if adv_val > enh_val else "Worse"
                elif key == 'consistency':
                    improvement = "Better" if adv_val < enh_val else "Worse"  # Lower is better
                else:
                    improvement = f"{adv_val/enh_val:.2f}x" if enh_val > 0 else "N/A"
                
                logger.info(f"{name:<20} {enh_val:<15.4f} {adv_val:<15.4f} {improvement:<12}")
    
    logger.info(f"\nComparison complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()