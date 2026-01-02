#!/usr/bin/env python3
"""
Convenience script to evaluate sampled backbones from flow matching.

This script automatically detects the standard sampling output structure:
  {samples_dir}/
    {scenario}/
      pdb/
        sample_XXXX.pdb
      angles/
        sample_XXXX.csv
      sampling_args.json (optional, contains motif regions)

Usage:
    python evaluations/evaluate_sampled_backbones.py \
        --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
        [--reference_dir data/references] \
        [--database_dir data/cath] \
        [--output_dir evaluation_results]
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

from evaluations.evaluate_pipeline import (
    run_comprehensive_evaluation,
    find_pdb_files,
    parse_motif_indices
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_sampling_scenarios(samples_dir: Path) -> List[Path]:
    """
    Find all sampling scenario directories.
    
    Looks for directories containing a 'pdb' subdirectory.
    
    Args:
        samples_dir: Root directory containing scenario subdirectories
    
    Returns:
        List of scenario directory paths
    """
    scenarios = []
    
    # Check if samples_dir itself contains pdb/ subdirectory
    if (samples_dir / "pdb").exists():
        scenarios.append(samples_dir)
    
    # Check subdirectories
    for item in samples_dir.iterdir():
        if item.is_dir() and (item / "pdb").exists():
            scenarios.append(item)
    
    return sorted(scenarios)


def find_saved_references(scenario_dir: Path) -> Optional[Path]:
    """
    Check if reference structures were automatically saved during sampling.
    
    Args:
        scenario_dir: Scenario directory path
    
    Returns:
        Path to references directory if found, None otherwise
    """
    ref_dir = scenario_dir / "references"
    if ref_dir.exists() and ref_dir.is_dir():
        # Check if there are any PDB files
        ref_files = list(ref_dir.glob("*.pdb")) + list(ref_dir.glob("reference_*"))
        if len(ref_files) > 0:
            logger.info(f"Found {len(ref_files)} automatically saved reference structures in {ref_dir}")
            return ref_dir
    return None


def extract_motif_regions_from_metadata(scenario_dir: Path) -> Optional[List[int]]:
    """
    Extract motif regions from sampling metadata if available.
    
    Checks for:
    - sampling_args.json
    - statistics.json
    - advanced_summary.json
    
    Args:
        scenario_dir: Scenario directory path
    
    Returns:
        List of motif indices or None if not found
    """
    metadata_files = [
        scenario_dir / "sampling_args.json",
        scenario_dir / "statistics.json",
        scenario_dir / "advanced_summary.json",
    ]
    
    for metadata_file in metadata_files:
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Try different possible keys
                motif_key = None
                for key in ['motif_regions', 'motif_regions_str', 'motif_indices']:
                    if key in data:
                        motif_key = key
                        break
                
                if motif_key:
                    motif_str = data[motif_key]
                    if isinstance(motif_str, str) and motif_str:
                        return parse_motif_indices(motif_str)
                    elif isinstance(motif_str, list):
                        # Flatten list of lists
                        indices = []
                        for item in motif_str:
                            if isinstance(item, (list, tuple)):
                                indices.extend(item)
                            else:
                                indices.append(item)
                        return sorted(indices)
            except Exception as e:
                logger.debug(f"Error reading {metadata_file}: {e}")
                continue
    
    return None


def evaluate_sampled_backbones(
    samples_dir: Path,
    reference_dir: Optional[Path] = None,
    database_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
    compute_rosetta: bool = False,
    compute_alphafold: bool = False,
    atom_type: str = "CA",
    n_threads: Optional[int] = None,
    evaluate_per_scenario: bool = True
) -> Dict:
    """
    Evaluate sampled backbones from flow matching.
    
    Args:
        samples_dir: Root directory containing sampled backbones
        reference_dir: Optional directory with reference structures
        database_dir: Optional directory with database for novelty
        output_dir: Output directory for evaluation results
        scenarios: Optional list of scenario names to evaluate (None = all)
        compute_rosetta: Whether to compute Rosetta energy
        compute_alphafold: Whether to compute AlphaFold2 confidence
        atom_type: Type of atoms to compare
        n_threads: Number of parallel threads
        evaluate_per_scenario: If True, evaluate each scenario separately
    
    Returns:
        Dictionary with evaluation results
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    # Find all scenarios
    all_scenarios = find_sampling_scenarios(samples_dir)
    logger.info(f"Found {len(all_scenarios)} sampling scenarios")
    
    if len(all_scenarios) == 0:
        raise ValueError(f"No sampling scenarios found in {samples_dir}")
    
    # Filter scenarios if specified
    if scenarios:
        all_scenarios = [s for s in all_scenarios if s.name in scenarios or s.stem in scenarios]
        logger.info(f"Filtered to {len(all_scenarios)} scenarios: {scenarios}")
    
    all_results = {}
    
    if evaluate_per_scenario:
        # Evaluate each scenario separately
        for scenario_dir in all_scenarios:
            logger.info("\n" + "="*80)
            logger.info(f"Evaluating scenario: {scenario_dir.name}")
            logger.info("="*80)
            
            # Find PDB files in this scenario
            pdb_dir = scenario_dir / "pdb"
            if not pdb_dir.exists():
                logger.warning(f"No pdb/ directory found in {scenario_dir}, skipping")
                continue
            
            # Extract motif regions from metadata
            motif_indices = extract_motif_regions_from_metadata(scenario_dir)
            if motif_indices:
                logger.info(f"Found motif regions in metadata: {motif_indices}")
            else:
                logger.info("No motif regions found in metadata")
            
            # Set output directory for this scenario
            if output_dir:
                scenario_output_dir = Path(output_dir) / scenario_dir.name
            else:
                scenario_output_dir = scenario_dir / "evaluation"
            
            # Check for automatically saved references
            auto_ref_dir = find_saved_references(scenario_dir)
            # Use auto-saved references if available, otherwise use provided reference_dir
            effective_ref_dir = auto_ref_dir if auto_ref_dir else reference_dir
            
            if auto_ref_dir:
                logger.info(f"Using automatically saved references from {auto_ref_dir}")
            
            # Run evaluation for this scenario
            try:
                scenario_results = run_comprehensive_evaluation(
                    generated_dir=pdb_dir,  # Point directly to pdb/ directory
                    reference_dir=effective_ref_dir,
                    output_dir=scenario_output_dir,
                    motif_indices=motif_indices,
                    database_dir=database_dir,
                    compute_rosetta=compute_rosetta,
                    compute_alphafold=compute_alphafold,
                    atom_type=atom_type,
                    n_threads=n_threads
                )
                all_results[scenario_dir.name] = scenario_results
                logger.info(f"✓ Evaluation complete for {scenario_dir.name}")
            except Exception as e:
                logger.error(f"Error evaluating {scenario_dir.name}: {e}")
                all_results[scenario_dir.name] = {"error": str(e)}
        
        # Create combined summary
        if output_dir:
            combined_output_dir = Path(output_dir)
        else:
            combined_output_dir = samples_dir / "evaluation_combined"
        
        combined_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        combined_summary = {
            "n_scenarios": len(all_scenarios),
            "scenarios": [s.name for s in all_scenarios],
            "evaluation_results": {}
        }
        
        for scenario_name, results in all_results.items():
            if "error" not in results:
                combined_summary["evaluation_results"][scenario_name] = {
                    "n_structures": results.get("n_generated_structures", 0),
                    "metrics_computed": list(results.keys())
                }
        
        with open(combined_output_dir / "combined_evaluation_summary.json", "w") as f:
            json.dump(combined_summary, f, indent=2)
        
        logger.info(f"\n✓ Combined summary saved to {combined_output_dir / 'combined_evaluation_summary.json'}")
    
    else:
        # Evaluate all scenarios together
        logger.info("Evaluating all scenarios together...")
        
        # Collect all PDB files from all scenarios
        all_pdb_files = []
        for scenario_dir in all_scenarios:
            pdb_dir = scenario_dir / "pdb"
            if pdb_dir.exists():
                pdbs = find_pdb_files(pdb_dir)
                all_pdb_files.extend(pdbs)
        
        if len(all_pdb_files) == 0:
            raise ValueError("No PDB files found in any scenario")
        
        logger.info(f"Found {len(all_pdb_files)} total PDB files across all scenarios")
        
        # Create temporary directory with symlinks or copy files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdb_dir = Path(tmpdir) / "all_pdbs"
            tmp_pdb_dir.mkdir()
            
            for pdb_file in all_pdb_files:
                # Create symlink with scenario prefix
                scenario_name = pdb_file.parent.parent.name
                new_name = f"{scenario_name}_{pdb_file.name}"
                (tmp_pdb_dir / new_name).symlink_to(pdb_file.resolve())
            
            # Run evaluation
            if output_dir:
                combined_output_dir = Path(output_dir)
            else:
                combined_output_dir = samples_dir / "evaluation_combined"
            
            all_results = run_comprehensive_evaluation(
                generated_dir=tmp_pdb_dir,
                reference_dir=reference_dir,
                output_dir=combined_output_dir,
                database_dir=database_dir,
                compute_rosetta=compute_rosetta,
                compute_alphafold=compute_alphafold,
                atom_type=atom_type,
                n_threads=n_threads
            )
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sampled backbones from flow matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all scenarios in a samples directory
  python evaluations/evaluate_sampled_backbones.py \\
      --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334

  # Evaluate specific scenarios
  python evaluations/evaluate_sampled_backbones.py \\
      --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \\
      --scenarios short_single_motif medium_single_motif

  # With reference structures
  python evaluations/evaluate_sampled_backbones.py \\
      --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \\
      --reference_dir data/references

  # With database for novelty
  python evaluations/evaluate_sampled_backbones.py \\
      --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \\
      --database_dir data/cath/dompdb
        """
    )
    
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Directory containing sampled backbones (with pdb/ subdirectories)"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Directory containing reference PDB structures (auto-detected if saved during sampling)"
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
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Specific scenarios to evaluate (default: all)"
    )
    parser.add_argument(
        "--compute_rosetta",
        action="store_true",
        help="Compute Rosetta energy scores"
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
        help="Number of parallel threads"
    )
    parser.add_argument(
        "--evaluate_together",
        action="store_true",
        help="Evaluate all scenarios together instead of separately"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_sampled_backbones(
        samples_dir=Path(args.samples_dir),
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        database_dir=Path(args.database_dir) if args.database_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        scenarios=args.scenarios,
        compute_rosetta=args.compute_rosetta,
        compute_alphafold=args.compute_alphafold,
        atom_type=args.atom_type,
        n_threads=args.n_threads,
        evaluate_per_scenario=not args.evaluate_together
    )
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

