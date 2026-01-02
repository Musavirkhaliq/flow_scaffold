#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline for protein generative models.

This script provides a unified interface to evaluate:
- Structural similarity (RMSD, TM-score, GDT)
- Motif recovery accuracy
- Sequence-structure compatibility
- Energy/physical plausibility
- Novelty and diversity

Usage:
    python evaluations/evaluate_pipeline.py \
        --generated_dir <path> \
        --reference_dir <path> \
        --output_dir <path> \
        [--motif_indices <indices>] \
        [--database_dir <path>] \
        [--compute_rosetta] \
        [--compute_alphafold]
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.structural_similarity import batch_evaluate_structural_similarity
from evaluations.motif_recovery import batch_evaluate_motif_recovery
from evaluations.sequence_structure_compatibility import batch_evaluate_sequence_structure_compatibility
from evaluations.energy_plausibility import batch_evaluate_energy_plausibility
from evaluations.novelty_diversity import evaluate_novelty_diversity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pdb_files(directory: Path, pattern: str = "*.pdb") -> List[Path]:
    """
    Find all PDB files in a directory.
    
    This function handles the standard sampling output structure:
    - Direct PDB files in the directory
    - PDB files in pdb/ subdirectory (standard sampling output)
    - PDB files in any subdirectory (recursive search)
    """
    pdbs = sorted(list(directory.glob(pattern)))
    if len(pdbs) == 0:
        # Check for standard pdb/ subdirectory (from sampling pipeline)
        pdb_subdir = directory / "pdb"
        if pdb_subdir.exists():
            pdbs = sorted(list(pdb_subdir.glob(pattern)))
        else:
            # Try recursive search in all subdirectories
            pdbs = sorted(list(directory.rglob(pattern)))
    return pdbs


def parse_motif_indices(indices_str: str) -> List[int]:
    """Parse motif indices from string (e.g., '10-20,30-40' or '10,11,12')."""
    indices = []
    for part in indices_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(indices)


def match_files(
    generated_files: List[Path],
    reference_files: List[Path],
    strategy: str = "name"
) -> List[tuple]:
    """
    Match generated files to reference files.
    
    Args:
        generated_files: List of generated file paths
        reference_files: List of reference file paths
        strategy: Matching strategy ('name', 'index', 'all_pairs')
    
    Returns:
        List of (generated, reference) tuples
    """
    if strategy == "name":
        # Match by filename (without extension)
        gen_dict = {f.stem: f for f in generated_files}
        ref_dict = {f.stem: f for f in reference_files}
        
        pairs = []
        for name in gen_dict:
            if name in ref_dict:
                pairs.append((gen_dict[name], ref_dict[name]))
        
        return pairs
    
    elif strategy == "index":
        # Match by index
        min_len = min(len(generated_files), len(reference_files))
        return list(zip(generated_files[:min_len], reference_files[:min_len]))
    
    elif strategy == "all_pairs":
        # All pairs (for diversity/novelty evaluation)
        pairs = []
        for gen in generated_files:
            for ref in reference_files:
                pairs.append((gen, ref))
        return pairs
    
    else:
        raise ValueError(f"Unknown matching strategy: {strategy}")


def run_comprehensive_evaluation(
    generated_dir: Path,
    reference_dir: Optional[Path] = None,
    output_dir: Path = None,
    motif_indices: Optional[List[int]] = None,
    motif_indices_list: Optional[List[List[int]]] = None,
    database_dir: Optional[Path] = None,
    compute_rosetta: bool = False,
    compute_alphafold: bool = False,
    atom_type: str = "CA",
    n_threads: Optional[int] = None
) -> Dict:
    """
    Run comprehensive evaluation pipeline.
    
    Args:
        generated_dir: Directory containing generated structures
        reference_dir: Optional directory containing reference structures
        output_dir: Directory to save evaluation results
        motif_indices: Optional motif indices (same for all pairs)
        motif_indices_list: Optional list of motif indices (one per pair)
        database_dir: Optional directory with database structures for novelty
        compute_rosetta: Whether to compute Rosetta energy
        compute_alphafold: Whether to compute AlphaFold2 confidence
        atom_type: Type of atoms to compare
        n_threads: Number of parallel threads
    
    Returns:
        Dictionary with all evaluation results
    """
    output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find PDB files
    logger.info("Finding PDB files...")
    generated_pdbs = find_pdb_files(generated_dir)
    logger.info(f"Found {len(generated_pdbs)} generated structures")
    
    if len(generated_pdbs) == 0:
        logger.error("No generated structures found!")
        return results
    
    # 1. Energy and Physical Plausibility (no reference needed)
    logger.info("\n" + "="*80)
    logger.info("1. Evaluating Energy and Physical Plausibility")
    logger.info("="*80)
    try:
        energy_results = batch_evaluate_energy_plausibility(
            generated_pdbs,
            compute_rosetta=compute_rosetta,
            compute_alphafold=compute_alphafold,
            n_threads=n_threads
        )
        energy_results.to_csv(output_dir / "energy_plausibility.csv", index=False)
        results["energy_plausibility"] = energy_results.to_dict('records')
        logger.info(f"✓ Energy/plausibility evaluation complete ({len(energy_results)} structures)")
    except Exception as e:
        logger.error(f"Error in energy/plausibility evaluation: {e}")
        results["energy_plausibility"] = {"error": str(e)}
    
    # 2. Novelty and Diversity (no reference needed, but database helps)
    logger.info("\n" + "="*80)
    logger.info("2. Evaluating Novelty and Diversity")
    logger.info("="*80)
    try:
        database_pdbs = None
        if database_dir:
            database_pdbs = find_pdb_files(database_dir)
            logger.info(f"Found {len(database_pdbs)} database structures for novelty comparison")
        
        novelty_results = evaluate_novelty_diversity(
            generated_pdbs,
            database_pdbs=database_pdbs,
            atom_type=atom_type,
            n_threads=n_threads
        )
        novelty_df = pd.DataFrame([novelty_results])
        novelty_df.to_csv(output_dir / "novelty_diversity.csv", index=False)
        results["novelty_diversity"] = novelty_results
        logger.info("✓ Novelty/diversity evaluation complete")
    except Exception as e:
        logger.error(f"Error in novelty/diversity evaluation: {e}")
        results["novelty_diversity"] = {"error": str(e)}
    
    # Reference-based evaluations
    if reference_dir:
        reference_pdbs = find_pdb_files(reference_dir)
        logger.info(f"Found {len(reference_pdbs)} reference structures")
        
        if len(reference_pdbs) == 0:
            logger.warning("No reference structures found, skipping reference-based evaluations")
        else:
            # Match files
            pairs = match_files(generated_pdbs, reference_pdbs, strategy="name")
            logger.info(f"Matched {len(pairs)} pairs for evaluation")
            
            if len(pairs) == 0:
                logger.warning("No matching pairs found, trying index-based matching")
                pairs = match_files(generated_pdbs, reference_pdbs, strategy="index")
                logger.info(f"Matched {len(pairs)} pairs using index-based matching")
            
            if len(pairs) > 0:
                query_pdbs, ref_pdbs = zip(*pairs)
                
                # 3. Structural Similarity
                logger.info("\n" + "="*80)
                logger.info("3. Evaluating Structural Similarity")
                logger.info("="*80)
                try:
                    # Prepare motif indices
                    motif_list = None
                    if motif_indices_list:
                        motif_list = motif_indices_list[:len(pairs)]
                    elif motif_indices:
                        motif_list = [motif_indices] * len(pairs)
                    
                    struct_results = batch_evaluate_structural_similarity(
                        list(query_pdbs),
                        list(ref_pdbs),
                        motif_indices_list=motif_list,
                        atom_type=atom_type,
                        n_threads=n_threads
                    )
                    struct_results.to_csv(output_dir / "structural_similarity.csv", index=False)
                    results["structural_similarity"] = struct_results.to_dict('records')
                    logger.info(f"✓ Structural similarity evaluation complete ({len(struct_results)} pairs)")
                except Exception as e:
                    logger.error(f"Error in structural similarity evaluation: {e}")
                    results["structural_similarity"] = {"error": str(e)}
                
                # 4. Motif Recovery (if motif indices provided)
                if motif_indices or motif_indices_list:
                    logger.info("\n" + "="*80)
                    logger.info("4. Evaluating Motif Recovery")
                    logger.info("="*80)
                    try:
                        motif_list = motif_indices_list[:len(pairs)] if motif_indices_list else [motif_indices] * len(pairs)
                        
                        motif_results = batch_evaluate_motif_recovery(
                            list(query_pdbs),
                            list(ref_pdbs),
                            motif_list,
                            atom_type=atom_type,
                            n_threads=n_threads
                        )
                        motif_results.to_csv(output_dir / "motif_recovery.csv", index=False)
                        results["motif_recovery"] = motif_results.to_dict('records')
                        logger.info(f"✓ Motif recovery evaluation complete ({len(motif_results)} pairs)")
                    except Exception as e:
                        logger.error(f"Error in motif recovery evaluation: {e}")
                        results["motif_recovery"] = {"error": str(e)}
                
                # 5. Sequence-Structure Compatibility
                logger.info("\n" + "="*80)
                logger.info("5. Evaluating Sequence-Structure Compatibility")
                logger.info("="*80)
                try:
                    seq_results = batch_evaluate_sequence_structure_compatibility(
                        list(query_pdbs),
                        list(ref_pdbs)
                    )
                    seq_results.to_csv(output_dir / "sequence_structure_compatibility.csv", index=False)
                    results["sequence_structure_compatibility"] = seq_results.to_dict('records')
                    logger.info(f"✓ Sequence-structure compatibility evaluation complete ({len(seq_results)} pairs)")
                except Exception as e:
                    logger.error(f"Error in sequence-structure compatibility evaluation: {e}")
                    results["sequence_structure_compatibility"] = {"error": str(e)}
    
    # Save summary
    summary = {
        "n_generated_structures": len(generated_pdbs),
        "n_reference_structures": len(reference_pdbs) if reference_dir else 0,
        "evaluation_metrics": list(results.keys()),
    }
    
    # Add summary statistics from each metric category
    if "energy_plausibility" in results and isinstance(results["energy_plausibility"], list):
        energy_df = pd.DataFrame(results["energy_plausibility"])
        if len(energy_df) > 0:
            summary["energy_plausibility"] = {}
            if "overall_quality_score" in energy_df.columns:
                summary["energy_plausibility"]["mean_quality_score"] = float(energy_df["overall_quality_score"].mean())
            if "ramachandran_favored" in energy_df.columns:
                summary["energy_plausibility"]["mean_ramachandran_favored"] = float(energy_df["ramachandran_favored"].mean())
                summary["energy_plausibility"]["mean_ramachandran_allowed"] = float(energy_df["ramachandran_allowed"].mean())
                summary["energy_plausibility"]["mean_ramachandran_outliers"] = float(energy_df["ramachandran_outliers"].mean())
            if "alpha_helix_fraction" in energy_df.columns:
                summary["energy_plausibility"]["mean_alpha_helix_fraction"] = float(energy_df["alpha_helix_fraction"].mean())
                summary["energy_plausibility"]["mean_beta_sheet_fraction"] = float(energy_df["beta_sheet_fraction"].mean())
            if "clash_rate" in energy_df.columns:
                summary["energy_plausibility"]["mean_clash_rate"] = float(energy_df["clash_rate"].mean())
                summary["energy_plausibility"]["mean_n_vdw_clashes"] = float(energy_df["n_vdw_clashes"].mean())
    
    if "novelty_diversity" in results:
        novelty = results["novelty_diversity"]
        if isinstance(novelty, dict):
            summary["novelty_diversity"] = {}
            if "mean_pairwise_rmsd" in novelty:
                summary["novelty_diversity"]["mean_pairwise_rmsd"] = float(novelty["mean_pairwise_rmsd"])
                summary["novelty_diversity"]["diversity_score"] = float(novelty.get("diversity_score", novelty.get("mean_pairwise_rmsd", 0)))
            if "mean_pairwise_tm" in novelty:
                summary["novelty_diversity"]["mean_pairwise_tm"] = float(novelty["mean_pairwise_tm"])
                summary["novelty_diversity"]["diversity_score_tm"] = float(novelty.get("diversity_score_tm", 0))
            if "novel_fraction" in novelty:
                summary["novelty_diversity"]["novel_fraction"] = float(novelty["novel_fraction"])
            if "seq_mean_pairwise_identity" in novelty:
                summary["novelty_diversity"]["seq_mean_pairwise_identity"] = float(novelty["seq_mean_pairwise_identity"])
                summary["novelty_diversity"]["seq_diversity_score"] = float(novelty.get("seq_diversity_score", 0))
    
    if "structural_similarity" in results and isinstance(results["structural_similarity"], list):
        struct_df = pd.DataFrame(results["structural_similarity"])
        if len(struct_df) > 0:
            summary["structural_similarity"] = {}
            if "RMSD" in struct_df.columns:
                summary["structural_similarity"]["mean_RMSD"] = float(struct_df["RMSD"].mean())
                summary["structural_similarity"]["median_RMSD"] = float(struct_df["RMSD"].median())
            if "TM_score" in struct_df.columns:
                summary["structural_similarity"]["mean_TM_score"] = float(struct_df["TM_score"].mean())
                summary["structural_similarity"]["median_TM_score"] = float(struct_df["TM_score"].median())
            if "GDT_TS" in struct_df.columns:
                summary["structural_similarity"]["mean_GDT_TS"] = float(struct_df["GDT_TS"].mean())
    
    if "motif_recovery" in results and isinstance(results["motif_recovery"], list):
        motif_df = pd.DataFrame(results["motif_recovery"])
        if len(motif_df) > 0:
            summary["motif_recovery"] = {}
            if "motif_RMSD" in motif_df.columns:
                summary["motif_recovery"]["mean_motif_RMSD"] = float(motif_df["motif_RMSD"].mean())
            if "motif_TM_score" in motif_df.columns:
                summary["motif_recovery"]["mean_motif_TM_score"] = float(motif_df["motif_TM_score"].mean())
            if "motif_preserved" in motif_df.columns:
                summary["motif_recovery"]["preservation_rate"] = float(motif_df["motif_preserved"].sum() / len(motif_df))
    
    if "sequence_structure_compatibility" in results and isinstance(results["sequence_structure_compatibility"], list):
        seq_df = pd.DataFrame(results["sequence_structure_compatibility"])
        if len(seq_df) > 0:
            summary["sequence_structure_compatibility"] = {}
            if "exact_recovery_rate" in seq_df.columns:
                summary["sequence_structure_compatibility"]["mean_exact_recovery"] = float(seq_df["exact_recovery_rate"].mean())
            if "similar_recovery_rate" in seq_df.columns:
                summary["sequence_structure_compatibility"]["mean_similar_recovery"] = float(seq_df["similar_recovery_rate"].mean())
            if "sequence_identity" in seq_df.columns:
                summary["sequence_structure_compatibility"]["mean_sequence_identity"] = float(seq_df["sequence_identity"].mean())
    
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - structural_similarity.csv")
    logger.info(f"  - motif_recovery.csv")
    logger.info(f"  - sequence_structure_compatibility.csv")
    logger.info(f"  - energy_plausibility.csv")
    logger.info(f"  - novelty_diversity.csv")
    logger.info(f"  - evaluation_summary.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation pipeline for protein generative models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluations/evaluate_pipeline.py \\
      --generated_dir results/samples \\
      --reference_dir data/references \\
      --output_dir evaluation_results

  # With motif indices
  python evaluations/evaluate_pipeline.py \\
      --generated_dir results/samples \\
      --reference_dir data/references \\
      --motif_indices "10-20,30-40" \\
      --output_dir evaluation_results

  # With database for novelty
  python evaluations/evaluate_pipeline.py \\
      --generated_dir results/samples \\
      --database_dir data/cath \\
      --output_dir evaluation_results
        """
    )
    
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory containing generated PDB structures"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Directory containing reference PDB structures"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--motif_indices",
        type=str,
        default=None,
        help="Motif indices (e.g., '10-20,30-40' or '10,11,12')"
    )
    parser.add_argument(
        "--motif_indices_file",
        type=str,
        default=None,
        help="JSON file with list of motif indices per structure"
    )
    parser.add_argument(
        "--database_dir",
        type=str,
        default=None,
        help="Directory with database structures for novelty evaluation"
    )
    parser.add_argument(
        "--compute_rosetta",
        action="store_true",
        help="Compute Rosetta energy scores (requires Rosetta installation)"
    )
    parser.add_argument(
        "--compute_alphafold",
        action="store_true",
        help="Compute AlphaFold2 confidence scores"
    )
    parser.add_argument(
        "--atom_type",
        type=str,
        default="CA",
        choices=["CA", "backbone"],
        help="Type of atoms to compare"
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=None,
        help="Number of parallel threads (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Parse motif indices
    motif_indices = None
    if args.motif_indices:
        motif_indices = parse_motif_indices(args.motif_indices)
    
    motif_indices_list = None
    if args.motif_indices_file:
        with open(args.motif_indices_file, 'r') as f:
            motif_indices_list = json.load(f)
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        generated_dir=Path(args.generated_dir),
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        output_dir=Path(args.output_dir),
        motif_indices=motif_indices,
        motif_indices_list=motif_indices_list,
        database_dir=Path(args.database_dir) if args.database_dir else None,
        compute_rosetta=args.compute_rosetta,
        compute_alphafold=args.compute_alphafold,
        atom_type=args.atom_type,
        n_threads=args.n_threads
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

