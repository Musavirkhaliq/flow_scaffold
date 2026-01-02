"""
Local motif recovery accuracy metrics.

Evaluates how well generated scaffolds preserve and recover target motifs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluations.structural_similarity import (
    compute_motif_rmsd,
    compute_motif_tm_score,
    compute_rmsd_from_pdb,
    compute_rmsd
)
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_motif_superposition_accuracy(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    atom_type: str = "CA"
) -> Dict[str, float]:
    """
    Compute motif superposition accuracy metrics.
    
    Evaluates how well the motif region aligns between query and reference.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        atom_type: Type of atoms to compare
    
    Returns:
        Dictionary with superposition metrics
    """
    query_pdb = Path(query_pdb)
    reference_pdb = Path(reference_pdb)
    
    if not query_pdb.exists() or not reference_pdb.exists():
        logger.error(f"PDB files not found: {query_pdb} or {reference_pdb}")
        return {}
    
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
            return {}
        
        coords1_motif = struct1.coord[motif_indices]
        coords2_motif = struct2.coord[motif_indices]
        
        # Compute RMSD after optimal superposition
        motif_rmsd = compute_rmsd(coords1_motif, coords2_motif, align=True)
        
        # Compute per-residue distances
        distances = np.sqrt(np.sum((coords1_motif - coords2_motif) ** 2, axis=1))
        
        # Compute statistics
        results = {
            "motif_RMSD": float(motif_rmsd),
            "motif_mean_distance": float(np.mean(distances)),
            "motif_median_distance": float(np.median(distances)),
            "motif_max_distance": float(np.max(distances)),
            "motif_std_distance": float(np.std(distances)),
            "motif_residues_within_1A": float(np.sum(distances <= 1.0) / len(distances)),
            "motif_residues_within_2A": float(np.sum(distances <= 2.0) / len(distances)),
            "motif_residues_within_5A": float(np.sum(distances <= 5.0) / len(distances)),
            "n_motif_residues": len(motif_indices),
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error computing motif superposition: {e}")
        return {}


def compute_motif_preservation_score(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    threshold_rmsd: float = 2.0
) -> Dict[str, float]:
    """
    Compute motif preservation score.
    
    A motif is considered "preserved" if its RMSD is below a threshold.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        threshold_rmsd: RMSD threshold in Angstroms (default: 2.0)
    
    Returns:
        Dictionary with preservation metrics
    """
    motif_rmsd = compute_motif_rmsd(query_pdb, reference_pdb, motif_indices)
    motif_tm = compute_motif_tm_score(query_pdb, reference_pdb, motif_indices)
    
    # Motif is preserved if RMSD < threshold and TM-score > 0.5
    is_preserved = (not np.isnan(motif_rmsd) and motif_rmsd < threshold_rmsd) and \
                  (not np.isnan(motif_tm) and motif_tm > 0.5)
    
    results = {
        "motif_RMSD": motif_rmsd,
        "motif_TM_score": motif_tm,
        "motif_preserved": bool(is_preserved),
        "preservation_threshold_rmsd": threshold_rmsd,
    }
    
    return results


def compute_scaffold_motif_interface_quality(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    interface_width: int = 3
) -> Dict[str, float]:
    """
    Evaluate the quality of the scaffold-motif interface.
    
    Analyzes residues near the motif boundary to assess how well
    the scaffold connects to the motif.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        interface_width: Number of residues on each side of motif boundary to analyze
    
    Returns:
        Dictionary with interface quality metrics
    """
    query_pdb = Path(query_pdb)
    reference_pdb = Path(reference_pdb)
    
    if not query_pdb.exists() or not reference_pdb.exists():
        logger.error(f"PDB files not found: {query_pdb} or {reference_pdb}")
        return {}
    
    try:
        # Read structures
        struct1 = PDBFile.read(str(query_pdb)).get_structure()[0]
        struct2 = PDBFile.read(str(reference_pdb)).get_structure()[0]
        
        # Get CA atoms
        struct1 = struct1[struct1.atom_name == "CA"]
        struct2 = struct2[struct2.atom_name == "CA"]
        
        motif_indices = np.array(sorted(motif_indices))
        motif_indices = motif_indices[motif_indices < len(struct1)]
        motif_indices = motif_indices[motif_indices < len(struct2)]
        
        if len(motif_indices) == 0:
            return {}
        
        # Define interface regions (residues near motif boundaries)
        min_motif = motif_indices.min()
        max_motif = motif_indices.max()
        
        # Left interface: residues before motif
        left_interface_start = max(0, min_motif - interface_width)
        left_interface_end = min_motif
        left_interface_indices = list(range(left_interface_start, left_interface_end))
        
        # Right interface: residues after motif
        right_interface_start = max_motif + 1
        right_interface_end = min(len(struct1), max_motif + 1 + interface_width)
        right_interface_indices = list(range(right_interface_start, right_interface_end))
        
        interface_indices = left_interface_indices + right_interface_indices
        
        if len(interface_indices) == 0:
            return {}
        
        # Extract interface coordinates
        coords1_interface = struct1.coord[interface_indices]
        coords2_interface = struct2.coord[interface_indices]
        
        # Compute interface RMSD
        interface_rmsd = compute_rmsd(coords1_interface, coords2_interface, align=True)
        
        # Compute distances
        distances = np.sqrt(np.sum((coords1_interface - coords2_interface) ** 2, axis=1))
        
        results = {
            "interface_RMSD": float(interface_rmsd),
            "interface_mean_distance": float(np.mean(distances)),
            "interface_max_distance": float(np.max(distances)),
            "n_interface_residues": len(interface_indices),
            "interface_width": interface_width,
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error computing interface quality: {e}")
        return {}


def evaluate_motif_recovery(
    query_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    motif_indices: List[int],
    atom_type: str = "CA",
    threshold_rmsd: float = 2.0,
    interface_width: int = 3
) -> Dict[str, float]:
    """
    Comprehensive motif recovery evaluation.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        motif_indices: List of residue indices (0-indexed) that form the motif
        atom_type: Type of atoms to compare
        threshold_rmsd: RMSD threshold for preservation (default: 2.0)
        interface_width: Width of interface region to analyze
    
    Returns:
        Dictionary with all motif recovery metrics
    """
    results = {}
    
    # Motif superposition accuracy
    superposition = compute_motif_superposition_accuracy(
        query_pdb, reference_pdb, motif_indices, atom_type=atom_type
    )
    results.update(superposition)
    
    # Motif preservation
    preservation = compute_motif_preservation_score(
        query_pdb, reference_pdb, motif_indices, threshold_rmsd=threshold_rmsd
    )
    results.update(preservation)
    
    # Interface quality
    interface = compute_scaffold_motif_interface_quality(
        query_pdb, reference_pdb, motif_indices, interface_width=interface_width
    )
    results.update(interface)
    
    return results


def batch_evaluate_motif_recovery(
    query_pdbs: List[Union[str, Path]],
    reference_pdbs: List[Union[str, Path]],
    motif_indices_list: List[List[int]],
    atom_type: str = "CA",
    threshold_rmsd: float = 2.0,
    interface_width: int = 3
) -> pd.DataFrame:
    """
    Batch evaluation of motif recovery for multiple pairs.
    
    Args:
        query_pdbs: List of query PDB file paths
        reference_pdbs: List of reference PDB file paths
        motif_indices_list: List of motif indices for each pair
        atom_type: Type of atoms to compare
        threshold_rmsd: RMSD threshold for preservation
        interface_width: Width of interface region
    
    Returns:
        DataFrame with evaluation results
    """
    assert len(query_pdbs) == len(reference_pdbs) == len(motif_indices_list), \
        "All lists must have same length"
    
    results = []
    for i, (query, ref, motif) in enumerate(zip(query_pdbs, reference_pdbs, motif_indices_list)):
        try:
            metrics = evaluate_motif_recovery(
                query, ref, motif,
                atom_type=atom_type,
                threshold_rmsd=threshold_rmsd,
                interface_width=interface_width
            )
            metrics["query"] = str(query)
            metrics["reference"] = str(ref)
            metrics["pair_index"] = i
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating pair {i}: {e}")
            results.append({
                "query": str(query),
                "reference": str(ref),
                "pair_index": i,
                "error": str(e)
            })
    
    return pd.DataFrame(results)

