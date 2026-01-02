"""
Structural similarity metrics for protein evaluation.

Implements:
- RMSD (Root Mean Square Deviation)
- TM-score (Template Modeling Score)
- GDT (Global Distance Test)
- Local motif RMSD
- Motif TM-score
"""

import os
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional TMalign import (requires TMalign binary in PATH)
try:
    from foldingdiff.tmalign import run_tmalign, max_tm_across_refs
    TMALIGN_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    TMALIGN_AVAILABLE = False
    # Create dummy functions if TMalign not available
    def run_tmalign(*args, **kwargs):
        logger.warning("TMalign not available. Install TMalign and add to PATH.")
        return np.nan
    def max_tm_across_refs(*args, **kwargs):
        logger.warning("TMalign not available. Install TMalign and add to PATH.")
        return np.nan, None

from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile


def compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True
) -> float:
    """
    Compute RMSD between two sets of coordinates.
    
    Args:
        coords1: Coordinates [N, 3]
        coords2: Coordinates [N, 3]
        align: If True, perform optimal superposition before computing RMSD
    
    Returns:
        RMSD value in Angstroms
    """
    assert coords1.shape == coords2.shape, "Coordinate arrays must have same shape"
    assert coords1.shape[1] == 3, "Coordinates must be 3D"
    
    # Remove NaN values
    valid_mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))
    coords1 = coords1[valid_mask]
    coords2 = coords2[valid_mask]
    
    if len(coords1) == 0:
        return np.nan
    
    # Center coordinates
    coords1_centered = coords1 - coords1.mean(axis=0)
    coords2_centered = coords2 - coords2.mean(axis=0)
    
    if align:
        # Optimal superposition using Procrustes analysis
        R, _ = orthogonal_procrustes(coords1_centered, coords2_centered)
        coords2_aligned = coords2_centered @ R.T
    else:
        coords2_aligned = coords2_centered
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean(np.sum((coords1_centered - coords2_aligned) ** 2, axis=1)))
    return float(rmsd)


def compute_rmsd_from_pdb(
    pdb1: Union[str, Path],
    pdb2: Union[str, Path],
    atom_type: str = "CA",
    chain: Optional[str] = None,
    align: bool = True
) -> float:
    """
    Compute RMSD between two PDB files.
    
    Args:
        pdb1: Path to first PDB file
        pdb2: Path to second PDB file
        atom_type: Type of atoms to compare (default: "CA" for C-alpha)
        chain: Specific chain to compare (None for all chains)
        align: If True, perform optimal superposition
    
    Returns:
        RMSD value in Angstroms
    """
    pdb1 = Path(pdb1)
    pdb2 = Path(pdb2)
    
    if not pdb1.exists() or not pdb2.exists():
        logger.error(f"PDB files not found: {pdb1} or {pdb2}")
        return np.nan
    
    try:
        # Read PDB files
        struct1 = PDBFile.read(str(pdb1)).get_structure()[0]
        struct2 = PDBFile.read(str(pdb2)).get_structure()[0]
        
        # Filter by chain if specified
        if chain is not None:
            struct1 = struct1[struct1.chain_id == chain]
            struct2 = struct2[struct2.chain_id == chain]
        
        # Get specified atom type
        if atom_type == "CA":
            struct1 = struct1[struct1.atom_name == "CA"]
            struct2 = struct2[struct2.atom_name == "CA"]
        elif atom_type == "backbone":
            struct1 = struct1[struc.filter_backbone(struct1)]
            struct2 = struct2[struc.filter_backbone(struct2)]
        else:
            struct1 = struct1[struct1.atom_name == atom_type]
            struct2 = struct2[struct2.atom_name == atom_type]
        
        # Extract coordinates
        coords1 = struct1.coord
        coords2 = struct2.coord
        
        # Ensure same length
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
        
        return compute_rmsd(coords1, coords2, align=align)
    
    except Exception as e:
        logger.error(f"Error computing RMSD: {e}")
        return np.nan


def compute_tm_score(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    fast: bool = False
) -> float:
    """
    Compute TM-score between two PDB files using TMalign.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        fast: Use fast mode (less accurate but faster)
    
    Returns:
        TM-score (0-1, higher is better)
    """
    query_pdb = Path(query_pdb)
    reference_pdb = Path(reference_pdb)
    
    if not query_pdb.exists() or not reference_pdb.exists():
        logger.error(f"PDB files not found: {query_pdb} or {reference_pdb}")
        return np.nan
    
    try:
        return run_tmalign(str(query_pdb), str(reference_pdb), fast=fast)
    except Exception as e:
        logger.error(f"Error computing TM-score: {e}")
        return np.nan


def compute_gdt(
    coords1: np.ndarray,
    coords2: np.ndarray,
    thresholds: List[float] = [1.0, 2.0, 4.0, 8.0],
    align: bool = True
) -> Dict[str, float]:
    """
    Compute Global Distance Test (GDT) scores.
    
    Args:
        coords1: Coordinates [N, 3]
        coords2: Coordinates [N, 3]
        thresholds: Distance thresholds in Angstroms (default: [1.0, 2.0, 4.0, 8.0])
        align: If True, perform optimal superposition
    
    Returns:
        Dictionary with GDT scores for each threshold and GDT_TS (average)
    """
    assert coords1.shape == coords2.shape
    assert coords1.shape[1] == 3
    
    # Remove NaN values
    valid_mask = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))
    coords1 = coords1[valid_mask]
    coords2 = coords2[valid_mask]
    
    if len(coords1) == 0:
        return {f"GDT_{t}": np.nan for t in thresholds}
    
    # Center and align if needed
    coords1_centered = coords1 - coords1.mean(axis=0)
    coords2_centered = coords2 - coords2.mean(axis=0)
    
    if align:
        R, _ = orthogonal_procrustes(coords1_centered, coords2_centered)
        coords2_aligned = coords2_centered @ R.T
    else:
        coords2_aligned = coords2_centered
    
    # Compute distances
    distances = np.sqrt(np.sum((coords1_centered - coords2_aligned) ** 2, axis=1))
    
    # Compute GDT for each threshold
    gdt_scores = {}
    for threshold in thresholds:
        fraction = np.sum(distances <= threshold) / len(distances)
        gdt_scores[f"GDT_{threshold}"] = float(fraction)
    
    # GDT_TS (Total Score) is the average of all thresholds
    gdt_scores["GDT_TS"] = float(np.mean(list(gdt_scores.values())))
    
    return gdt_scores


def compute_motif_rmsd(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    atom_type: str = "CA"
) -> float:
    """
    Compute RMSD specifically for motif region.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        atom_type: Type of atoms to compare
    
    Returns:
        Motif RMSD in Angstroms
    """
    query_pdb = Path(query_pdb)
    reference_pdb = Path(reference_pdb)
    
    if not query_pdb.exists() or not reference_pdb.exists():
        logger.error(f"PDB files not found: {query_pdb} or {reference_pdb}")
        return np.nan
    
    try:
        # Read structures
        struct1 = PDBFile.read(str(query_pdb)).get_structure()[0]
        struct2 = PDBFile.read(str(reference_pdb)).get_structure()[0]
        
        # Get specified atom type
        if atom_type == "CA":
            struct1 = struct1[struct1.atom_name == "CA"]
            struct2 = struct2[struct2.atom_name == "CA"]
        elif atom_type == "backbone":
            struct1 = struct1[struc.filter_backbone(struct1)]
            struct2 = struct2[struc.filter_backbone(struct2)]
        
        # Extract motif coordinates
        motif_indices = np.array(motif_indices)
        motif_indices = motif_indices[motif_indices < len(struct1)]
        motif_indices = motif_indices[motif_indices < len(struct2)]
        
        if len(motif_indices) == 0:
            logger.warning("No valid motif indices")
            return np.nan
        
        coords1_motif = struct1.coord[motif_indices]
        coords2_motif = struct2.coord[motif_indices]
        
        return compute_rmsd(coords1_motif, coords2_motif, align=True)
    
    except Exception as e:
        logger.error(f"Error computing motif RMSD: {e}")
        return np.nan


def compute_motif_tm_score(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    fast: bool = False
) -> float:
    """
    Compute TM-score for motif region only.
    
    This extracts the motif region from both structures and computes TM-score.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        fast: Use fast mode for TMalign
    
    Returns:
        Motif TM-score (0-1)
    """
    query_pdb = Path(query_pdb)
    reference_pdb = Path(reference_pdb)
    
    if not query_pdb.exists() or not reference_pdb.exists():
        logger.error(f"PDB files not found: {query_pdb} or {reference_pdb}")
        return np.nan
    
    try:
        # Read structures
        struct1 = PDBFile.read(str(query_pdb)).get_structure()[0]
        struct2 = PDBFile.read(str(reference_pdb)).get_structure()[0]
        
        # Extract motif regions
        motif_indices = np.array(motif_indices)
        motif_indices = motif_indices[motif_indices < len(struct1)]
        motif_indices = motif_indices[motif_indices < len(struct2)]
        
        if len(motif_indices) == 0:
            logger.warning("No valid motif indices")
            return np.nan
        
        struct1_motif = struct1[motif_indices]
        struct2_motif = struct2[motif_indices]
        
        # Write temporary PDB files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
            
            temp_pdb1 = f1.name
            temp_pdb2 = f2.name
            
            PDBFile.write(temp_pdb1, struct1_motif)
            PDBFile.write(temp_pdb2, struct2_motif)
        
        # Compute TM-score
        tm_score = run_tmalign(temp_pdb1, temp_pdb2, fast=fast)
        
        # Clean up
        os.unlink(temp_pdb1)
        os.unlink(temp_pdb2)
        
        return tm_score
    
    except Exception as e:
        logger.error(f"Error computing motif TM-score: {e}")
        return np.nan


def evaluate_structural_similarity(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: Optional[List[int]] = None,
    atom_type: str = "CA"
) -> Dict[str, float]:
    """
    Comprehensive structural similarity evaluation.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: Optional list of motif residue indices
        atom_type: Type of atoms to compare
    
    Returns:
        Dictionary with all structural similarity metrics
    """
    results = {}
    
    # Overall structural similarity
    results["RMSD"] = compute_rmsd_from_pdb(query_pdb, reference_pdb, atom_type=atom_type)
    results["TM_score"] = compute_tm_score(query_pdb, reference_pdb, fast=False)
    
    # GDT scores
    try:
        struct1 = PDBFile.read(str(query_pdb)).get_structure()[0]
        struct2 = PDBFile.read(str(reference_pdb)).get_structure()[0]
        
        if atom_type == "CA":
            struct1 = struct1[struct1.atom_name == "CA"]
            struct2 = struct2[struct2.atom_name == "CA"]
        elif atom_type == "backbone":
            struct1 = struct1[struc.filter_backbone(struct1)]
            struct2 = struct2[struc.filter_backbone(struct2)]
        
        min_len = min(len(struct1), len(struct2))
        coords1 = struct1.coord[:min_len]
        coords2 = struct2.coord[:min_len]
        
        gdt_scores = compute_gdt(coords1, coords2)
        results.update(gdt_scores)
    except Exception as e:
        logger.warning(f"Could not compute GDT scores: {e}")
    
    # Motif-specific metrics
    if motif_indices is not None:
        results["motif_RMSD"] = compute_motif_rmsd(
            query_pdb, reference_pdb, motif_indices, atom_type=atom_type
        )
        results["motif_TM_score"] = compute_motif_tm_score(
            query_pdb, reference_pdb, motif_indices, fast=False
        )
    
    return results


def batch_evaluate_structural_similarity(
    query_pdbs: List[Union[str, Path]],
    reference_pdbs: List[Union[str, Path]],
    motif_indices_list: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
    atom_type: str = "CA"
) -> pd.DataFrame:
    """
    Batch evaluation of structural similarity for multiple pairs.
    
    Args:
        query_pdbs: List of query PDB file paths
        reference_pdbs: List of reference PDB file paths
        motif_indices_list: Optional list of motif indices for each pair
        n_threads: Number of parallel threads (default: CPU count)
        atom_type: Type of atoms to compare
    
    Returns:
        DataFrame with evaluation results
    """
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    assert len(query_pdbs) == len(reference_pdbs), "Query and reference lists must have same length"
    
    if motif_indices_list is not None:
        assert len(motif_indices_list) == len(query_pdbs), "Motif indices list must match query list length"
    
    def evaluate_pair(args):
        idx, query, ref, motif = args
        try:
            results = evaluate_structural_similarity(
                query, ref, motif_indices=motif, atom_type=atom_type
            )
            results["query"] = str(query)
            results["reference"] = str(ref)
            results["pair_index"] = idx
            return results
        except Exception as e:
            logger.error(f"Error evaluating pair {idx}: {e}")
            return {"query": str(query), "reference": str(ref), "pair_index": idx, "error": str(e)}
    
    # Prepare arguments
    args_list = []
    for i, (q, r) in enumerate(zip(query_pdbs, reference_pdbs)):
        motif = motif_indices_list[i] if motif_indices_list is not None else None
        args_list.append((i, q, r, motif))
    
    # Parallel evaluation
    if n_threads > 1 and len(args_list) > 1:
        with mp.Pool(n_threads) as pool:
            results = pool.map(evaluate_pair, args_list)
    else:
        results = [evaluate_pair(args) for args in args_list]
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

