#!/usr/bin/env python3
"""
Quick sampling script from checkpoint
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertConfig

from foldingdiff.enhanced_models import BertForFlowMatchingEnhanced
from foldingdiff import angles_and_coords

# Configuration
CHECKPOINT = "/home/musa/Documents/augment-projects/foldingdiff/flow_scaf/results/enhanced_flow/enhanced_flow_251230_131020/models/best_by_valid/epoch=047-val_loss=0.1154.ckpt"
OUTPUT_DIR = "flow_scaf/samples/test_trained_model"
N_SAMPLES = 10
LENGTH = 80
NUM_STEPS = 100
DEVICE = "cuda:0"

print("=" * 80)
print("SAMPLING FROM TRAINED MODEL")
print("=" * 80)
print(f"Checkpoint: {CHECKPOINT}")
print(f"Samples: {N_SAMPLES} × length {LENGTH}")
print(f"Steps: {NUM_STEPS}")
print()

# Create output directories
os.makedirs(f"{OUTPUT_DIR}/pdb", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/angles", exist_ok=True)

# Load checkpoint
print("Loading checkpoint...")
try:
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
except:
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = checkpoint["state_dict"]

# Create model config - MUST match checkpoint training config
config = BertConfig(
    vocab_size=30522,
    hidden_size=384,
    num_hidden_layers=12,  # Checkpoint has 12 layers (0-11)
    num_attention_heads=6,
    intermediate_size=1536,
    max_position_embeddings=128,  # Checkpoint trained with max_seq_len=128
)

# Create model - MUST match checkpoint embedding config
model = BertForFlowMatchingEnhanced(
    config=config,
    ft_is_angular=[True, True, True, False, False, False],
    ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
    use_enhanced_embedding=True,
    embedding_config={
        'use_sequence': True,  # Checkpoint includes sequence embeddings
        'use_coords': True,
        'use_local_frames': True,
        'use_pairwise': True,
        'use_secondary_structure': True,  # Checkpoint includes SS embeddings
    },
    use_motif_conditioning=True,
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✓ Model loaded\n")

# Sample
print("Generating samples...")
print("=" * 80)

def sample_flow_matching(model, length, num_steps, device):
    """Simple Euler sampling for flow matching"""
    with torch.no_grad():
        # Start from noise
        x = torch.randn(1, length, 6, device=device)
        
        # Empty motif mask (unconditional)
        motif_mask = torch.zeros(1, length, 1, device=device)
        motif_features = torch.zeros(1, length, 6, device=device)
        
        # Attention mask (all valid)
        attention_mask = torch.ones(1, length, device=device)
        
        # Dummy inputs for sequence and secondary structure (model expects them)
        # Use glycine (aa_type=7) as neutral amino acid
        aa_types = torch.full((1, length), 7, dtype=torch.long, device=device)
        # Uniform secondary structure (coil)
        secondary_structure = torch.zeros(1, length, 3, device=device)
        secondary_structure[:, :, 2] = 1.0  # All coil
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.ones(1, device=device) * (step / num_steps)
            
            # Get velocity prediction
            v_pred = model(
                x,
                timestep=t,
                attention_mask=attention_mask,
                aa_types=aa_types,
                secondary_structure=secondary_structure,
                motif_mask=motif_mask,
                motif_features=motif_features,
            )
            
            # Euler step
            x = x + v_pred * dt
            
            # Wrap angular features to [-π, π]
            x[:, :, :3] = torch.atan2(torch.sin(x[:, :, :3]), torch.cos(x[:, :, :3]))
        
        return x[0].cpu()

samples = []
for i in tqdm(range(N_SAMPLES)):
    sample = sample_flow_matching(model, LENGTH, NUM_STEPS, DEVICE)
    samples.append(sample)

print(f"\n✓ Generated {len(samples)} samples\n")

# Save and convert to PDB
print("Converting to PDB structures...")
print("=" * 80)

for i, sample in enumerate(tqdm(samples)):
    # Save angles
    np.savetxt(
        f"{OUTPUT_DIR}/angles/sample_{i:04d}.csv",
        sample.numpy(),
        delimiter=",",
        header="phi,psi,omega,tau,CA:C:1N,C:1N:1CA",
        comments=""
    )
    
    # Add means back to angles
    sample_corrected = sample.clone()
    sample_corrected[:, 2] += np.pi  # omega: add 180°
    sample_corrected[:, 3] += 1.92   # tau: add ~110°
    sample_corrected[:, 4] += 2.01   # CA:C:1N: add ~115°
    sample_corrected[:, 5] += 2.11   # C:1N:1CA: add ~121°
    
    # Create dataframe
    angles_df = pd.DataFrame({
        'phi': sample_corrected[:, 0].numpy(),
        'psi': sample_corrected[:, 1].numpy(),
        'omega': sample_corrected[:, 2].numpy(),
        'tau': sample_corrected[:, 3].numpy(),
        'CA:C:1N': sample_corrected[:, 4].numpy(),
        'C:1N:1CA': sample_corrected[:, 5].numpy(),
        '0C:1N': 1.329,
        'N:CA': 1.458,
        'CA:C': 1.525,
    })
    
    # Convert to PDB
    pdb_file = f"{OUTPUT_DIR}/pdb/sample_{i:04d}.pdb"
    angles_and_coords.create_new_chain_nerf(
        pdb_file,
        angles_df,
        angles_to_set=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
        dists_to_set=['0C:1N', 'N:CA', 'CA:C'],
    )

print(f"\n✓ Saved PDB files to {OUTPUT_DIR}/pdb")
print(f"✓ Saved angles to {OUTPUT_DIR}/angles")
print("\n" + "=" * 80)
print("SAMPLING COMPLETE!")
print("=" * 80)
print(f"Output: {OUTPUT_DIR}")
print("\nNext: Run ProteinMPNN with proper temperature (0.3-0.5)")
print(f"  bash run_proteinmpnn.sh --input_dir {OUTPUT_DIR}/pdb --temperature 0.3")
print("\nNote: Temperature 0.1 causes repetitive sequences!")
print("      Use 0.3 for balance, 0.5 for diversity")
print("=" * 80)
