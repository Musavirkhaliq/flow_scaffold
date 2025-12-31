#!/usr/bin/env python3
"""
Convert angle CSV files to PDB structures for visualization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from foldingdiff import angles_and_coords


def angles_to_pdb(angles_file: Path, output_pdb: Path):
    """Convert angles CSV to PDB file"""
    # Load angles
    angles = np.loadtxt(angles_file, delimiter=",", skiprows=1)
    
    # Remove padding (rows that are all zeros)
    non_zero_mask = ~np.all(angles == 0, axis=1)
    angles = angles[non_zero_mask]
    
    if len(angles) == 0:
        print(f"Warning: No non-zero angles in {angles_file}")
        return False
    
    # Create DataFrame with proper column names
    angle_names = angles_and_coords.EXHAUSTIVE_ANGLES[:6]
    df = pd.DataFrame(angles, columns=angle_names)
    
    # CRITICAL: Add means back to angles
    # The model learned mean-centered angles, so we need to add the means back
    df['omega'] += np.pi  # Add 180° for trans peptide bonds
    df['tau'] += 1.92  # Add ~110° in radians
    df['CA:C:1N'] += 2.01  # Add ~115° in radians
    df['C:1N:1CA'] += 2.11  # Add ~121° in radians
    
    # Add required distance columns for create_new_chain_nerf
    # Use standard peptide bond lengths
    df['0C:1N'] = 1.329  # C-N peptide bond
    df['N:CA'] = 1.458   # N-CA bond
    df['CA:C'] = 1.525   # CA-C bond
    
    try:
        # Create PDB using create_new_chain_nerf
        angles_and_coords.create_new_chain_nerf(
            str(output_pdb),
            df,
            angles_to_set=angle_names,
            dists_to_set=angles_and_coords.EXHAUSTIVE_DISTS
        )
        return True
    except Exception as e:
        print(f"Error converting {angles_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert angle CSV files to PDB")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing angle CSV files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for PDB files")
    parser.add_argument("--pattern", type=str, default="*.csv",
                       help="File pattern to match")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting angles from: {input_dir}")
    print(f"Output PDB files to: {output_dir}")
    
    # Find all CSV files
    csv_files = sorted(input_dir.glob(args.pattern))
    
    if not csv_files:
        print(f"No CSV files found matching {args.pattern}")
        return
    
    print(f"Found {len(csv_files)} files to convert")
    
    success_count = 0
    for csv_file in csv_files:
        pdb_file = output_dir / csv_file.stem.replace("sample_", "structure_")
        pdb_file = pdb_file.with_suffix(".pdb")
        
        print(f"Converting {csv_file.name} -> {pdb_file.name}...", end=" ")
        
        if angles_to_pdb(csv_file, pdb_file):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    print(f"\nConversion complete: {success_count}/{len(csv_files)} successful")
    print(f"PDB files saved to: {output_dir}")


if __name__ == "__main__":
    main()
