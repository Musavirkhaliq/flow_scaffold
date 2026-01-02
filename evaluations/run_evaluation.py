#!/usr/bin/env python3
"""
Simple script to run comprehensive evaluation on a samples directory.

Usage:
    python evaluations/run_evaluation.py <samples_dir>
    
Example:
    python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.evaluate_sampled_backbones import evaluate_sampled_backbones
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation on sampled backbones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage - just provide the samples directory
  python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234
  
  # With reference structures
  python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234 \\
      --reference_dir data/references
  
  # With database for novelty
  python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234 \\
      --database_dir data/cath/dompdb
        """
    )
    
    parser.add_argument(
        "samples_dir",
        type=str,
        help="Directory containing sampled backbones"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Directory containing reference PDB structures"
    )
    parser.add_argument(
        "--database_dir",
        type=str,
        default=None,
        help="Directory with database structures for novelty evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: {samples_dir}/evaluation)"
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
    samples_dir = Path(args.samples_dir)
    
    if not samples_dir.exists():
        logger.error(f"Error: Samples directory not found: {samples_dir}")
        logger.error("Usage: python evaluations/run_evaluation.py <samples_dir>")
        return 1
    
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    database_dir = Path(args.database_dir) if args.database_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    logger.info("="*80)
    logger.info("RUNNING COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    logger.info(f"Samples directory: {samples_dir}")
    if reference_dir:
        logger.info(f"Reference directory: {reference_dir}")
    if database_dir:
        logger.info(f"Database directory: {database_dir}")
    logger.info("")
    
    # Run evaluation
    try:
        results = evaluate_sampled_backbones(
            samples_dir=samples_dir,
            reference_dir=reference_dir,
            database_dir=database_dir,
            output_dir=output_dir,
            compute_rosetta=args.compute_rosetta,
            compute_alphafold=args.compute_alphafold,
            atom_type=args.atom_type,
            n_threads=args.n_threads,
            evaluate_per_scenario=True
        )
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*80)
        
        if output_dir:
            eval_output = Path(output_dir)
        else:
            eval_output = samples_dir / "evaluation"
        
        logger.info(f"\nResults saved to: {eval_output}")
        logger.info("\nGenerated files:")
        logger.info("  - energy_plausibility.csv")
        logger.info("  - novelty_diversity.csv")
        if reference_dir:
            logger.info("  - structural_similarity.csv")
            logger.info("  - sequence_structure_compatibility.csv")
        logger.info("  - evaluation_summary.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

