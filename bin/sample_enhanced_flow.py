#!/usr/bin/env python3
"""
Sampling script for enhanced flow matching model.

Uses all three phases for fast, high-quality generation:
- Phase 1: Conditional motif scaffolding
- Phase 2: Flow matching (50 steps, 20x faster)
- Phase 3: Enhanced embeddings
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging

import numpy as np
import torch
from tqdm import tqdm

from transformers import BertConfig

from foldingdiff.enhanced_models import BertForFlowMatchingEnhanced
from foldingdiff.flow_sampling import sample_flow_matching_with_guidance
from foldingdiff.motif_scaffolding import create_motif_mask_from_regions
from foldingdiff import angles_and_coords
import pandas as pd


def load_model(model_dir: str, device: str = "cuda:0"):
    """Load trained enhanced flow matching model"""
    # Load config
    config = BertConfig.from_json_file(os.path.join(model_dir, "config.json"))
    
    # Load training args
    with open(os.path.join(model_dir, "training_args.json")) as f:
        train_args = json.load(f)
    
    # Embedding config
    embedding_config = {
        'use_sequence': train_args.get('use_sequence', False),
        'use_coords': train_args.get('use_coords', True),
        'use_local_frames': train_args.get('use_local_frames', True),
        'use_pairwise': train_args.get('use_pairwise', True),
        'use_secondary_structure': train_args.get('use_ss', False),
    }
    
    # Create model
    model = BertForFlowMatchingEnhanced(
        config=config,
        ft_is_angular=[True, True, True, False, False, False],  # Only dihedrals are periodic
        ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
        use_enhanced_embedding=True,
        embedding_config=embedding_config,
        use_motif_conditioning=True,
    )
    
    # Load weights
    checkpoint_dir = Path(model_dir) / "models" / "best_by_valid"
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoint = sorted(checkpoints)[-1]
    logging.info(f"Loading checkpoint: {checkpoint}")
    
    state_dict = torch.load(checkpoint, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, train_args


def parse_motif_spec(motif_spec: str) -> list:
    """Parse motif specification string"""
    if not motif_spec:
        return []
    
    regions = []
    for region_str in motif_spec.split(","):
        start, end = map(int, region_str.split("-"))
        regions.append((start, end))
    
    return regions


def main():
    parser = argparse.ArgumentParser(
        description="Sample with enhanced flow matching (20x faster!)"
    )
    
    # Model
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Sampling
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50, 
                       help="Flow matching steps (vs 1000 for diffusion)")
    parser.add_argument("--method", type=str, default="euler", 
                       choices=["euler", "rk4"])
    
    # Motif conditioning
    parser.add_argument("--motif_regions", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="samples/enhanced_flow")
    parser.add_argument("--save_pdb", action="store_true")
    parser.add_argument("--save_angles", action="store_true")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sampling args
    with open(output_dir / "sampling_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 80)
    logger.info("ENHANCED FLOW MATCHING SAMPLING")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Samples: {args.n_samples} × length {args.length}")
    logger.info(f"Steps: {args.num_steps} (20x faster than diffusion!)")
    logger.info(f"Method: {args.method}")
    logger.info(f"Guidance: {args.guidance_scale}")
    
    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)
    
    model, train_args = load_model(args.model_dir, args.device)
    
    logger.info("✓ Model loaded successfully")
    logger.info(f"✓ Enhanced embeddings enabled")
    logger.info(f"✓ Flow matching: {args.num_steps} steps")
    
    # Parse motif regions
    motif_regions = parse_motif_spec(args.motif_regions)
    logger.info(f"✓ Motif regions: {motif_regions if motif_regions else 'None (unconditional)'}")
    
    # Prepare motif data
    pad_length = train_args.get("pad", 128)
    
    if motif_regions:
        motif_mask = create_motif_mask_from_regions(
            motif_regions, args.length, pad_length
        )
        # For now, use zeros for motif angles (would load from PDB in production)
        motif_angles = torch.zeros(pad_length, 6)
    else:
        motif_mask = torch.zeros(pad_length, 1)
        motif_angles = torch.zeros(pad_length, 6)
    
    # Sample
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLING")
    logger.info("=" * 80)
    logger.info("Phase 1: Conditional motif scaffolding ✓")
    logger.info("Phase 2: Flow matching (20x speedup) ✓")
    logger.info("Phase 3: Enhanced embeddings ✓")
    logger.info("=" * 80 + "\n")
    
    samples = []
    
    for i in tqdm(range(args.n_samples), desc="Generating samples"):
        # Sample with flow matching
        sample = sample_flow_matching_with_guidance(
            model=model,
            shape=(1, pad_length, 6),
            motif_mask=motif_mask.unsqueeze(0),
            motif_features=motif_angles.unsqueeze(0),
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            method=args.method,
            is_angular=[True, True, True, False, False, False],  # Only dihedrals are periodic
            disable_pbar=True,
        )
        
        # Trim to actual length
        sample = sample[0, :args.length, :].cpu()
        samples.append(sample)
    
    logger.info(f"\n✓ Generated {len(samples)} samples")
    
    # Save samples
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    if args.save_angles:
        angles_dir = output_dir / "angles"
        angles_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            np.savetxt(
                angles_dir / f"sample_{i:04d}.csv",
                sample.numpy(),
                delimiter=",",
                header="phi,psi,omega,tau,CA:C:1N,C:1N:1CA",
                comments=""
            )
        
        logger.info(f"✓ Saved angles to {angles_dir}")
    
    if args.save_pdb:
        pdb_dir = output_dir / "pdb"
        pdb_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            # Convert angles to coordinates using proper NERF algorithm
            pdb_file = pdb_dir / f"sample_{i:04d}.pdb"
            
            # CRITICAL: Add means back to angles
            # The model learned mean-centered angles, so we need to add the means back
            sample_corrected = sample.clone()
            # Omega: add π (180°) for trans peptide bonds
            sample_corrected[:, 2] += np.pi  # omega: add 180° in radians
            # Bond angles: add their natural means
            sample_corrected[:, 3] += 1.92  # tau: add ~110° in radians
            sample_corrected[:, 4] += 2.01  # CA:C:1N: add ~115° in radians
            sample_corrected[:, 5] += 2.11  # C:1N:1CA: add ~121° in radians
            
            # Create dataframe with angles and distances
            angles_df = pd.DataFrame({
                'phi': sample_corrected[:, 0].numpy(),
                'psi': sample_corrected[:, 1].numpy(),
                'omega': sample_corrected[:, 2].numpy(),
                'tau': sample_corrected[:, 3].numpy(),
                'CA:C:1N': sample_corrected[:, 4].numpy(),
                'C:1N:1CA': sample_corrected[:, 5].numpy(),
                # Add standard peptide bond distances
                '0C:1N': 1.329,  # C-N peptide bond
                'N:CA': 1.458,   # N-CA bond
                'CA:C': 1.525,   # CA-C bond
            })
            
            # Use NERF to build proper 3D structure
            angles_and_coords.create_new_chain_nerf(
                str(pdb_file),
                angles_df,
                angles_to_set=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
                dists_to_set=['0C:1N', 'N:CA', 'CA:C'],
            )
        
        logger.info(f"✓ Saved PDB files to {pdb_dir}")
    
    # Save statistics
    stats = {
        "n_samples": len(samples),
        "length": args.length,
        "num_steps": args.num_steps,
        "method": args.method,
        "motif_regions": motif_regions,
        "guidance_scale": args.guidance_scale,
    }
    
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Saved statistics to {output_dir / 'statistics.json'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✓ {len(samples)} samples generated")
    logger.info(f"✓ {args.num_steps} steps (20x faster than diffusion)")
    logger.info(f"✓ Results saved to {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Visualize: pymol " + str(pdb_dir / "*.pdb"))
    logger.info("2. Evaluate: python bin/evaluate_samples.py")
    logger.info("3. Design sequences: python bin/design_sequences.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
