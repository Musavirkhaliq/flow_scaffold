#!/usr/bin/env python3
"""
Validate generated protein structures using OmegaFold predictions.

This script:
1. Takes designed sequences from ProteinMPNN
2. Predicts structures using OmegaFold
3. Computes structural metrics (TM-score, RMSD)
4. Generates validation report
"""
import sys
import argparse
import subprocess
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple
import biotite.structure as struc
import biotite.structure.io.pdb as pdb


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse FASTA file and return sequences"""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def run_omegafold(sequence: str, output_path: Path) -> bool:
    """Run OmegaFold prediction for a sequence"""
    # Create temp fasta
    temp_fasta = output_path.parent / f"{output_path.stem}_temp.fasta"
    with open(temp_fasta, 'w') as f:
        f.write(f">sequence\n{sequence}\n")
    
    # Use conda run to activate omegafold environment
    cmd = [
        "conda", "run", "-n", "omegafold",
        "omegafold",
        str(temp_fasta),
        str(output_path.parent),
        "--model", "2",
        "--num_cycle", "10"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        temp_fasta.unlink()
        
        # OmegaFold creates output with specific naming
        predicted_pdb = output_path.parent / "sequence.pdb"
        if predicted_pdb.exists():
            predicted_pdb.rename(output_path)
            return True
        return False
    except Exception as e:
        print(f"Error running OmegaFold: {e}")
        if temp_fasta.exists():
            temp_fasta.unlink()
        return False


def compute_rmsd(pdb1_path: Path, pdb2_path: Path) -> float:
    """Compute RMSD between two structures using biotite"""
    try:
        # Load structures
        pdb1_file = pdb.PDBFile.read(str(pdb1_path))
        struct1 = pdb1_file.get_structure()[0]
        
        pdb2_file = pdb.PDBFile.read(str(pdb2_path))
        struct2 = pdb2_file.get_structure()[0]
        
        # Get CA atoms
        ca1 = struct1[struct1.atom_name == "CA"]
        ca2 = struct2[struct2.atom_name == "CA"]
        
        # Ensure same length
        min_len = min(len(ca1), len(ca2))
        ca1 = ca1[:min_len]
        ca2 = ca2[:min_len]
        
        # Superimpose and compute RMSD
        superimposed, _ = struc.superimpose(ca1, ca2)
        rmsd = struc.rmsd(ca1, superimposed)
        
        return float(rmsd)
    except Exception as e:
        print(f"Error computing RMSD: {e}")
        return -1.0


def compute_tm_score(pdb1_path: Path, pdb2_path: Path) -> Tuple[float, float]:
    """
    Compute TM-score between two structures using TMalign.
    Returns (tm_score1, tm_score2) normalized by each structure length.
    """
    try:
        cmd = ["TMalign", str(pdb1_path), str(pdb2_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return -1.0, -1.0
        
        # Parse output
        tm1, tm2 = -1.0, -1.0
        for line in result.stdout.split('\n'):
            if line.startswith('TM-score='):
                parts = line.split()
                if 'Chain_1' in line or 'first protein' in line.lower():
                    tm1 = float(parts[1])
                elif 'Chain_2' in line or 'second protein' in line.lower():
                    tm2 = float(parts[1])
        
        return tm1, tm2
    except Exception as e:
        print(f"Error computing TM-score: {e}")
        return -1.0, -1.0


def validate_structure(
    original_pdb: Path,
    sequence: str,
    output_dir: Path,
    structure_name: str
) -> Dict:
    """Validate a single structure"""
    print(f"  Validating {structure_name}...")
    
    # Predict structure with OmegaFold
    predicted_pdb = output_dir / f"{structure_name}_predicted.pdb"
    
    print(f"    Running OmegaFold...")
    if not run_omegafold(sequence, predicted_pdb):
        print(f"    ✗ OmegaFold prediction failed")
        return {
            'structure': structure_name,
            'sequence_length': len(sequence),
            'omegafold_success': False,
            'rmsd': -1.0,
            'tm_score': -1.0,
            'tm_score_norm': -1.0
        }
    
    print(f"    ✓ Structure predicted")
    
    # Compute metrics
    print(f"    Computing RMSD...")
    rmsd = compute_rmsd(original_pdb, predicted_pdb)
    
    print(f"    Computing TM-score...")
    tm1, tm2 = compute_tm_score(original_pdb, predicted_pdb)
    
    results = {
        'structure': structure_name,
        'sequence_length': len(sequence),
        'omegafold_success': True,
        'rmsd': rmsd,
        'tm_score': tm1,  # Normalized by original structure
        'tm_score_norm': tm2,  # Normalized by predicted structure
        'predicted_pdb': str(predicted_pdb)
    }
    
    print(f"    RMSD: {rmsd:.3f} Å")
    print(f"    TM-score: {tm1:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated structures with OmegaFold"
    )
    parser.add_argument(
        "--sequences_dir",
        type=str,
        required=True,
        help="Directory containing ProteinMPNN sequences"
    )
    parser.add_argument(
        "--structures_dir",
        type=str,
        required=True,
        help="Directory containing original PDB structures"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validation",
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Maximum number of structures to validate"
    )
    parser.add_argument(
        "--sequences_per_structure",
        type=int,
        default=3,
        help="Number of sequences to validate per structure"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("STRUCTURE VALIDATION WITH OMEGAFOLD")
    print("="*80)
    print()
    
    sequences_dir = Path(args.sequences_dir)
    structures_dir = Path(args.structures_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence directories
    seq_subdirs = [d for d in sequences_dir.iterdir() if d.is_dir()]
    
    if not seq_subdirs:
        print(f"No sequence directories found in {sequences_dir}")
        return
    
    if args.max_structures:
        seq_subdirs = seq_subdirs[:args.max_structures]
    
    print(f"Found {len(seq_subdirs)} structures to validate")
    print()
    
    all_results = []
    
    for i, seq_dir in enumerate(seq_subdirs, 1):
        struct_name = seq_dir.name
        print(f"[{i}/{len(seq_subdirs)}] Processing {struct_name}")
        print("-" * 80)
        
        # Find original PDB
        original_pdb = structures_dir / f"{struct_name}.pdb"
        if not original_pdb.exists():
            print(f"  ✗ Original PDB not found: {original_pdb}")
            continue
        
        # Find sequence files (ProteinMPNN puts them in seqs/ subdirectory)
        fasta_files = list(seq_dir.glob("*.fa"))
        if not fasta_files:
            # Try seqs subdirectory
            seqs_dir = seq_dir / "seqs"
            if seqs_dir.exists():
                fasta_files = list(seqs_dir.glob("*.fa"))
        
        if not fasta_files:
            print(f"  ✗ No FASTA files found in {seq_dir}")
            continue
        
        # Parse sequences
        sequences = {}
        for fasta_file in fasta_files:
            seqs = parse_fasta(fasta_file)
            sequences.update(seqs)
        
        if not sequences:
            print(f"  ✗ No sequences found")
            continue
        
        print(f"  Found {len(sequences)} designed sequences")
        
        # Validate top N sequences
        seq_items = list(sequences.items())[:args.sequences_per_structure]
        
        for seq_id, sequence in seq_items:
            result = validate_structure(
                original_pdb,
                sequence,
                output_dir,
                f"{struct_name}_{seq_id}"
            )
            all_results.append(result)
        
        print()
    
    # Save results
    results_file = output_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute summary statistics
    successful = [r for r in all_results if r['omegafold_success']]
    
    if successful:
        rmsds = [r['rmsd'] for r in successful if r['rmsd'] > 0]
        tm_scores = [r['tm_score'] for r in successful if r['tm_score'] > 0]
        
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Total structures validated: {len(all_results)}")
        print(f"Successful predictions: {len(successful)}")
        print()
        
        if rmsds:
            print(f"RMSD Statistics:")
            print(f"  Mean: {np.mean(rmsds):.3f} Å")
            print(f"  Median: {np.median(rmsds):.3f} Å")
            print(f"  Std: {np.std(rmsds):.3f} Å")
            print(f"  Min: {np.min(rmsds):.3f} Å")
            print(f"  Max: {np.max(rmsds):.3f} Å")
            print()
        
        if tm_scores:
            print(f"TM-score Statistics:")
            print(f"  Mean: {np.mean(tm_scores):.3f}")
            print(f"  Median: {np.median(tm_scores):.3f}")
            print(f"  Std: {np.std(tm_scores):.3f}")
            print(f"  Min: {np.min(tm_scores):.3f}")
            print(f"  Max: {np.max(tm_scores):.3f}")
            print()
            
            # Quality thresholds
            high_quality = sum(1 for tm in tm_scores if tm >= 0.5)
            medium_quality = sum(1 for tm in tm_scores if 0.3 <= tm < 0.5)
            low_quality = sum(1 for tm in tm_scores if tm < 0.3)
            
            print(f"Quality Distribution:")
            print(f"  High (TM ≥ 0.5): {high_quality} ({100*high_quality/len(tm_scores):.1f}%)")
            print(f"  Medium (0.3 ≤ TM < 0.5): {medium_quality} ({100*medium_quality/len(tm_scores):.1f}%)")
            print(f"  Low (TM < 0.3): {low_quality} ({100*low_quality/len(tm_scores):.1f}%)")
        
        print()
        print(f"Results saved to: {results_file}")
        print("="*80)


if __name__ == "__main__":
    main()
