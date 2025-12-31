#!/usr/bin/env python3
"""
End-to-end training script for enhanced flow matching model.

Combines all three phases:
- Phase 1: Conditional motif scaffolding
- Phase 2: Flow matching (20x faster sampling)
- Phase 3: Enhanced embeddings (9x richer features)
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import BertConfig

from foldingdiff.enhanced_datasets import create_enhanced_dataset
from foldingdiff.enhanced_models import BertForFlowMatchingEnhancedTraining


def main():
    parser = argparse.ArgumentParser(
        description="Train enhanced flow matching model (all 3 phases)"
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/cath")
    parser.add_argument("--pad", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=40)
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    
    # Enhanced features
    parser.add_argument("--use_coords", action="store_true", default=True)
    parser.add_argument("--use_local_frames", action="store_true", default=True)
    parser.add_argument("--use_pairwise", action="store_true", default=True)
    parser.add_argument("--use_sequence", action="store_true", default=False)
    parser.add_argument("--use_ss", action="store_true", default=False)
    
    # Motif scaffolding
    parser.add_argument("--motif_length_min", type=int, default=5)
    parser.add_argument("--motif_length_max", type=int, default=20)
    parser.add_argument("--motif_prob", type=float, default=0.8)
    parser.add_argument("--max_motifs", type=int, default=1)
    parser.add_argument("--guidance_dropout", type=float, default=0.1)
    
    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    
    # Flow matching
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)  # Increased from 5e-5
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_scheduler", type=str, default="LinearWarmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/enhanced_flow")
    parser.add_argument("--experiment_name", type=str, default="full_system")
    
    # System
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 80)
    logger.info("ENHANCED FLOW MATCHING TRAINING")
    logger.info("All 3 Phases: Conditional + Flow Matching + Enhanced Embeddings")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    
    # Create datasets
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    
    train_dset = create_enhanced_dataset(
        split="train" if not args.toy else None,
        pad=args.pad,
        min_length=args.min_length,
        toy=50 if args.toy else 0,
        compute_coords=args.use_coords,
        compute_ss=args.use_ss,
        use_motif_scaffolding=True,
        motif_length_range=(args.motif_length_min, args.motif_length_max),
        motif_prob=args.motif_prob,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
    )
    
    val_dset = create_enhanced_dataset(
        split="validation" if not args.toy else None,
        pad=args.pad,
        min_length=args.min_length,
        toy=10 if args.toy else 0,
        compute_coords=args.use_coords,
        compute_ss=args.use_ss,
        use_motif_scaffolding=True,
        motif_length_range=(args.motif_length_min, args.motif_length_max),
        motif_prob=args.motif_prob,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
    )
    
    logger.info(f"✓ Train dataset: {len(train_dset)} examples")
    logger.info(f"✓ Val dataset: {len(val_dset)} examples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    
    config = BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.pad,
    )
    
    # Save config
    config.save_pretrained(output_dir)
    
    # Embedding configuration``
    embedding_config = {
        'use_sequence': args.use_sequence,
        'use_coords': args.use_coords,
        'use_local_frames': args.use_local_frames,
        'use_pairwise': args.use_pairwise,
        'use_secondary_structure': args.use_ss,
    }
    
    model = BertForFlowMatchingEnhancedTraining(
        config=config,
        ft_is_angular=[True, True, True, False, False, False],  # Only dihedrals are periodic
        ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
        use_enhanced_embedding=True,
        embedding_config=embedding_config,
        use_motif_conditioning=True,
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        lr_scheduler=args.lr_scheduler,
        guidance_dropout=args.guidance_dropout,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model created: {num_params:,} parameters")
    logger.info(f"✓ Enhanced embeddings: {embedding_config}")
    logger.info(f"✓ Flow matching: {args.timesteps} timesteps, {args.beta_schedule} schedule")
    logger.info(f"✓ Motif scaffolding: prob={args.motif_prob}, length={args.motif_length_min}-{args.motif_length_max}")
    
    # Set up callbacks
    logger.info("\n" + "=" * 80)
    logger.info("SETTING UP TRAINING")
    logger.info("=" * 80)
    
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=output_dir / "models" / "best_by_train",
        filename="epoch={epoch:03d}-train_loss={train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    
    checkpoint_callback_val = ModelCheckpoint(
        dirpath=output_dir / "models" / "best_by_valid",
        filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=args.experiment_name,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback_train, checkpoint_callback_val, lr_monitor],
        logger=tb_logger,
        gradient_clip_val=args.gradient_clip,
        log_every_n_steps=10,
        val_check_interval=1.0,
        accumulate_grad_batches=1,  # Can increase if memory is limited
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision for speed
    )
    
    logger.info(f"✓ Trainer configured")
    logger.info(f"✓ Epochs: {args.epochs}")
    logger.info(f"✓ Batch size: {args.batch_size}")
    logger.info(f"✓ Learning rate: {args.lr}")
    logger.info(f"✓ GPUs: {args.gpus if torch.cuda.is_available() else 0}")
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info("Phase 1: Conditional motif scaffolding ✓")
    logger.info("Phase 2: Flow matching (20x speedup) ✓")
    logger.info("Phase 3: Enhanced embeddings (9x richer) ✓")
    logger.info("=" * 80 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✓ Models saved to: {output_dir / 'models'}")
    logger.info(f"✓ Logs saved to: {output_dir / 'logs'}")
    logger.info(f"✓ Config saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Monitor training: tensorboard --logdir " + str(output_dir / "logs"))
    logger.info("2. Sample structures: python bin/sample_enhanced_flow.py")
    logger.info("3. Evaluate quality: python bin/evaluate_samples.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
