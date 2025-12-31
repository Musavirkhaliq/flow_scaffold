#!/usr/bin/env python3
"""
Design sequences for generated backbones using ProteinMPNN.

This script takes backbone structures and designs amino acid sequences
that are likely to fold into those structures.
"""
import sys
from pathlib import Path
import subprocess
import json
import argparse


def check_proteinmpnn_installed():
    """Check if ProteinMPNN is available"""
    # Check common locations
    possible_paths = [
        Path("/home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN"),  # Absolute path
        Path("../software/ProteinMPNN"),  # From flow_scaf
        Path("../../software/ProteinMPNN"),  # From flow_scaf/bin
        Path("software/ProteinMPNN"),
        Path("ProteinMPNN"),
    ]
    
    for mpnn_dir in possible_paths:
        if mpnn_dir.exists() and (mpnn_dir / "protein_mpnn_run.py").exists():
            print(f"✓ ProteinMPNN found at: {mpnn_dir}")
            return mpnn_dir
    
    print("✗ ProteinMPNN not found in expected locations:")
    for p in possible_paths:
        print(f"  - {p}")
    print("\nPlease install ProteinMPNN:")
    print("  git clone https://github.com/dauparas/ProteinMPNN.git software/ProteinMPNN")
    
    return None


def design_sequences(
    pdb_path: Path,
    output_dir: Path,
    mpnn_dir: Path,
    num_sequences: int = 10,
    temperature: float = 0.1,
    fixed_positions: str = None
):
    """
    Design sequences for a backbone structure.
    
    Args:
        pdb_path: Path to PDB file
        output_dir: Output directory for sequences
        mpnn_dir: Path to ProteinMPNN installation
        num_sequences: Number of sequences to design
        temperature: Sampling temperature (lower = more conservative)
        fixed_positions: Positions to keep fixed (for motif regions)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ProteinMPNN command
    mpnn_script = mpnn_dir / "protein_mpnn_run.py"
    
    cmd = [
        "python", str(mpnn_script),
        "--ca_only",  # Use CA-only mode for backbone structures
        "--pdb_path", str(pdb_path.absolute()),
        "--out_folder", str(output_dir.absolute()),
        "--num_seq_per_target", str(num_sequences),
        "--sampling_temp", str(temperature),
        "--seed", "42",
        "--batch_size", "1",
    ]
    
    if fixed_positions:
        cmd.extend(["--fixed_positions", fixed_positions])
    
    print(f"Designing sequences for: {pdb_path.name}")
    print(f"Output: {output_dir}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Error designing sequences:")
        print(result.stderr)
        return False
    
    # Check if sequences were generated (ProteinMPNN puts them in seqs/ subdirectory)
    seqs_dir = output_dir / "seqs"
    fasta_files = list(seqs_dir.glob("*.fa")) if seqs_dir.exists() else []
    
    if not fasta_files:
        # Also check root directory
        fasta_files = list(output_dir.glob("*.fa"))
    
    if fasta_files:
        print(f"✓ Generated {len(fasta_files)} sequence files")
        return True
    else:
        print("✗ No sequences generated")
        print(f"   Checked: {output_dir} and {seqs_dir}")
        if result.stdout:
            print(f"   stdout: {result.stdout[-200:]}")
        if result.stderr:
            print(f"   stderr: {result.stderr[-200:]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Design sequences for backbone structures using ProteinMPNN"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing PDB files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sequences",
        help="Output directory for designed sequences"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10,
        help="Number of sequences to design per structure"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (0.1 = conservative, 0.3 = balanced, 0.5-1.0 = diverse)"
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Maximum number of structures to process"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PROTEINMPNN SEQUENCE DESIGN")
    print("="*80)
    print()
    
    # Check ProteinMPNN installation
    mpnn_dir = check_proteinmpnn_installed()
    if not mpnn_dir:
        print("✗ ProteinMPNN not found")
        return
    
    print()
    
    # Find PDB files
    input_dir = Path(args.input_dir)
    pdb_files = sorted(input_dir.glob("*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return
    
    if args.max_structures:
        pdb_files = pdb_files[:args.max_structures]
    
    print(f"Found {len(pdb_files)} structures to process")
    print()
    
    # Process each structure
    output_base = Path(args.output_dir)
    
    success_count = 0
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] Processing {pdb_file.name}")
        print("-" * 80)
        
        # Create output directory for this structure
        struct_name = pdb_file.stem
        output_dir = output_base / struct_name
        
        # Design sequences
        if design_sequences(
            pdb_file,
            output_dir,
            mpnn_dir,
            num_sequences=args.num_sequences,
            temperature=args.temperature
        ):
            success_count += 1
            print(f"✓ Sequences designed for {pdb_file.name}")
        else:
            print(f"✗ Failed to design sequences for {pdb_file.name}")
        
        print()
    
    print("="*80)
    print("SEQUENCE DESIGN COMPLETE")
    print("="*80)
    print(f"Successfully processed: {success_count}/{len(pdb_files)}")
    print(f"Sequences saved to: {output_base}")
    print()
    print("Next steps:")
    print("1. Review designed sequences")
    print("2. Predict structures with AlphaFold2")
    print("3. Compare predicted vs designed structures")
    print("="*80)


if __name__ == "__main__":
    main()
