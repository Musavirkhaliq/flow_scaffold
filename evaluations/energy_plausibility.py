"""
Energy and physical plausibility evaluation metrics.

Evaluates physical plausibility using:
- Rosetta energy scores (if available)
- AlphaFold2 confidence metrics (pLDDT)
- Van der Waals clash detection
- Ramachandran plot analysis
"""

import logging
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import multiprocessing as mp

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
from foldingdiff.vdw_clashes import count_clashes
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

# Optional lDDT import (requires Docker setup)
# Note: lDDT is not currently used in this module, but kept for future use
try:
    from foldingdiff.lddt import lddt
    LDDT_AVAILABLE = True
except (ImportError, AssertionError, FileNotFoundError) as e:
    LDDT_AVAILABLE = False
    logger.debug(f"lDDT not available: {e}")


def compute_ramachandran_quality(
    pdb_path: Union[str, Path]
) -> Dict[str, float]:
    """
    Evaluate Ramachandran plot quality.
    
    Computes the fraction of residues in allowed/favored regions.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Dictionary with Ramachandran metrics
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        return {}
    
    try:
        struct = PDBFile.read(str(pdb_path)).get_structure()[0]
        
        # Get dihedral angles
        phi, psi, omega = struc.dihedral_backbone(struct)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(phi) | np.isnan(psi))
        phi = phi[valid_mask]
        psi = psi[valid_mask]
        
        if len(phi) == 0:
            return {}
        
        # Define Ramachandran regions (in radians)
        # Favored regions
        alpha_helix = (phi > -2.0) & (phi < -0.5) & (psi > -1.5) & (psi < 0.5)
        beta_sheet = (phi > -2.5) & (phi < -0.5) & (psi > 1.0) & (psi < 2.5)
        left_handed_alpha = (phi > 0.3) & (phi < 1.2) & (psi > 0.3) & (psi < 1.2)
        
        # Allowed regions (broader)
        allowed_alpha = (phi > -2.5) & (phi < -0.3) & (psi > -2.0) & (psi < 1.0)
        allowed_beta = (phi > -2.8) & (phi < -0.3) & (psi > 0.5) & (psi < 3.0)
        
        favored = alpha_helix | beta_sheet | left_handed_alpha
        allowed = allowed_alpha | allowed_beta
        
        # Omega analysis (peptide bond)
        omega_valid = omega[valid_mask]
        trans_peptide = np.abs(omega_valid) > np.pi / 2  # Trans is ~180 degrees
        
        results = {
            "n_residues": len(phi),
            "ramachandran_favored": float(np.sum(favored) / len(phi)),
            "ramachandran_allowed": float(np.sum(allowed) / len(phi)),
            "ramachandran_outliers": float(np.sum(~allowed) / len(phi)),
            "alpha_helix_fraction": float(np.sum(alpha_helix) / len(phi)),
            "beta_sheet_fraction": float(np.sum(beta_sheet) / len(phi)),
            "left_handed_alpha_fraction": float(np.sum(left_handed_alpha) / len(phi)),
            "trans_peptide_fraction": float(np.sum(trans_peptide) / len(omega_valid)) if len(omega_valid) > 0 else 0.0,
            "cis_peptide_fraction": float(np.sum(~trans_peptide) / len(omega_valid)) if len(omega_valid) > 0 else 0.0,
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error computing Ramachandran quality: {e}")
        return {}


def compute_vdw_clash_score(
    pdb_path: Union[str, Path],
    alpha: float = 0.63
) -> Dict[str, float]:
    """
    Compute van der Waals clash score.
    
    Args:
        pdb_path: Path to PDB file
        alpha: Clash threshold factor (default: 0.63)
    
    Returns:
        Dictionary with clash metrics
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        return {}
    
    try:
        n_clashes = count_clashes(str(pdb_path), alpha=alpha)
        
        # Get number of atoms for normalization
        struct = PDBFile.read(str(pdb_path)).get_structure()[0]
        n_atoms = len(struct)
        
        results = {
            "n_vdw_clashes": int(n_clashes),
            "n_atoms": n_atoms,
            "clash_rate": float(n_clashes / n_atoms) if n_atoms > 0 else 0.0,
            "clash_alpha": alpha,
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error computing VDW clashes: {e}")
        return {}


def compute_rosetta_energy(
    pdb_path: Union[str, Path],
    rosetta_bin: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute Rosetta energy score (if Rosetta is available).
    
    Args:
        pdb_path: Path to PDB file
        rosetta_bin: Path to Rosetta binaries (searches PATH if None)
    
    Returns:
        Dictionary with energy metrics (empty if Rosetta not available)
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        return {}
    
    # Check for Rosetta
    if rosetta_bin is None:
        rosetta_bin = shutil.which("score_jd2") or shutil.which("score")
    
    if rosetta_bin is None:
        logger.warning("Rosetta not found in PATH. Skipping Rosetta energy calculation.")
        return {"rosetta_available": False}
    
    try:
        # Run Rosetta scoring
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sc', delete=False) as f:
            score_file = f.name
        
        cmd = [rosetta_bin, "-s", str(pdb_path), "-out:file:scorefile", score_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.warning(f"Rosetta scoring failed: {result.stderr}")
            return {"rosetta_available": True, "rosetta_error": result.stderr}
        
        # Parse score file
        # Rosetta score files are tab-separated with headers
        if os.path.exists(score_file):
            try:
                df = pd.read_csv(score_file, sep='\t')
                if len(df) > 0:
                    results = {
                        "rosetta_available": True,
                        "total_score": float(df.iloc[0].get('total_score', np.nan)),
                        "fa_atr": float(df.iloc[0].get('fa_atr', np.nan)),  # Attractive
                        "fa_rep": float(df.iloc[0].get('fa_rep', np.nan)),  # Repulsive
                        "fa_sol": float(df.iloc[0].get('fa_sol', np.nan)),  # Solvation
                        "fa_elec": float(df.iloc[0].get('fa_elec', np.nan)),  # Electrostatic
                    }
                    os.unlink(score_file)
                    return results
            except Exception as e:
                logger.warning(f"Error parsing Rosetta score file: {e}")
                if os.path.exists(score_file):
                    os.unlink(score_file)
        
        return {"rosetta_available": True, "rosetta_parse_error": True}
    
    except subprocess.TimeoutExpired:
        logger.warning("Rosetta scoring timed out")
        return {"rosetta_available": True, "rosetta_timeout": True}
    except Exception as e:
        logger.warning(f"Error running Rosetta: {e}")
        return {"rosetta_available": True, "rosetta_error": str(e)}


def compute_alphafold2_confidence(
    pdb_path: Union[str, Path],
    alphafold_bin: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute AlphaFold2 confidence metrics (pLDDT).
    
    Note: This requires AlphaFold2 to be installed or pLDDT scores
    to be pre-computed and stored in B-factor column.
    
    Args:
        pdb_path: Path to PDB file
        alphafold_bin: Path to AlphaFold2 binary (optional)
    
    Returns:
        Dictionary with confidence metrics
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        return {}
    
    try:
        struct = PDBFile.read(str(pdb_path)).get_structure()[0]
        
        # Check if B-factor column contains pLDDT scores
        # AlphaFold2 typically stores pLDDT in B-factor column
        if hasattr(struct, 'b_factor') and struct.b_factor is not None:
            plddt_scores = struct.b_factor
            
            # Filter valid scores
            valid_scores = plddt_scores[~np.isnan(plddt_scores)]
            
            if len(valid_scores) > 0:
                results = {
                    "plddt_mean": float(np.mean(valid_scores)),
                    "plddt_median": float(np.median(valid_scores)),
                    "plddt_std": float(np.std(valid_scores)),
                    "plddt_min": float(np.min(valid_scores)),
                    "plddt_max": float(np.max(valid_scores)),
                    "plddt_high_confidence": float(np.sum(valid_scores >= 90) / len(valid_scores)),
                    "plddt_medium_confidence": float(np.sum((valid_scores >= 70) & (valid_scores < 90)) / len(valid_scores)),
                    "plddt_low_confidence": float(np.sum(valid_scores < 70) / len(valid_scores)),
                    "n_residues": len(valid_scores),
                }
                return results
        
        # If no B-factor data, try to run AlphaFold2 (if available)
        if alphafold_bin is None:
            alphafold_bin = shutil.which("alphafold")
        
        if alphafold_bin is None:
            logger.warning("AlphaFold2 not found and no pLDDT in B-factor column")
            return {"alphafold_available": False}
        
        # Note: Running AlphaFold2 is computationally expensive
        # This is a placeholder - actual implementation would require
        # AlphaFold2 setup and configuration
        logger.warning("AlphaFold2 inference not implemented in this version")
        return {"alphafold_available": True, "alphafold_not_implemented": True}
    
    except Exception as e:
        logger.error(f"Error computing AlphaFold2 confidence: {e}")
        return {}


def evaluate_energy_plausibility(
    pdb_path: Union[str, Path],
    compute_rosetta: bool = True,
    compute_alphafold: bool = False,
    rosetta_bin: Optional[str] = None
) -> Dict[str, float]:
    """
    Comprehensive energy and physical plausibility evaluation.
    
    Args:
        pdb_path: Path to PDB file
        compute_rosetta: Whether to compute Rosetta energy (if available)
        compute_alphafold: Whether to compute AlphaFold2 confidence
        rosetta_bin: Path to Rosetta binary
    
    Returns:
        Dictionary with all plausibility metrics
    """
    results = {}
    
    # Ramachandran quality
    ramachandran = compute_ramachandran_quality(pdb_path)
    results.update(ramachandran)
    
    # VDW clashes
    vdw = compute_vdw_clash_score(pdb_path)
    results.update(vdw)
    
    # Rosetta energy (if requested and available)
    if compute_rosetta:
        rosetta = compute_rosetta_energy(pdb_path, rosetta_bin=rosetta_bin)
        results.update(rosetta)
    
    # AlphaFold2 confidence (if requested)
    if compute_alphafold:
        af2 = compute_alphafold2_confidence(pdb_path)
        results.update(af2)
    
    # Overall quality score (composite)
    quality_score = 0.0
    if "ramachandran_favored" in results:
        quality_score += results["ramachandran_favored"] * 0.4
    if "clash_rate" in results:
        quality_score += (1.0 - min(results["clash_rate"] * 10, 1.0)) * 0.3
    if "plddt_mean" in results:
        quality_score += (results["plddt_mean"] / 100.0) * 0.3
    
    results["overall_quality_score"] = float(quality_score)
    
    return results


# Module-level function for multiprocessing (must be at module level for pickle)
def _evaluate_single_energy(pdb_path_and_args):
    """Helper function for multiprocessing - must be at module level."""
    pdb_path, compute_rosetta, compute_alphafold, rosetta_bin = pdb_path_and_args
    try:
        metrics = evaluate_energy_plausibility(
            pdb_path,
            compute_rosetta=compute_rosetta,
            compute_alphafold=compute_alphafold,
            rosetta_bin=rosetta_bin
        )
        metrics["pdb_path"] = str(pdb_path)
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating {pdb_path}: {e}")
        return {"pdb_path": str(pdb_path), "error": str(e)}


def batch_evaluate_energy_plausibility(
    pdb_paths: List[Union[str, Path]],
    compute_rosetta: bool = True,
    compute_alphafold: bool = False,
    rosetta_bin: Optional[str] = None,
    n_threads: Optional[int] = None
) -> pd.DataFrame:
    """
    Batch evaluation of energy and physical plausibility.
    
    Args:
        pdb_paths: List of PDB file paths
        compute_rosetta: Whether to compute Rosetta energy
        compute_alphafold: Whether to compute AlphaFold2 confidence
        rosetta_bin: Path to Rosetta binary
        n_threads: Number of parallel threads
    
    Returns:
        DataFrame with evaluation results
    """
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # Prepare arguments for multiprocessing
    args_list = [
        (pdb_path, compute_rosetta, compute_alphafold, rosetta_bin)
        for pdb_path in pdb_paths
    ]
    
    # Parallel evaluation
    if n_threads > 1 and len(pdb_paths) > 1:
        with mp.Pool(n_threads) as pool:
            results = pool.map(_evaluate_single_energy, args_list)
    else:
        results = [_evaluate_single_energy(args) for args in args_list]
    
    return pd.DataFrame(results)

