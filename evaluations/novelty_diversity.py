"""
Novelty and diversity metrics for protein generation evaluation.

Evaluates:
- Structural diversity among generated samples
- Novelty compared to reference database (e.g., CATH)
- Sequence diversity
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from evaluations.structural_similarity import compute_tm_score, compute_rmsd_from_pdb, compute_rmsd
from evaluations.sequence_structure_compatibility import compute_sequence_diversity, extract_sequence_from_pdb

# Optional TMalign import (requires TMalign binary in PATH)
try:
    from foldingdiff.tmalign import max_tm_across_refs
    TMALIGN_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    TMALIGN_AVAILABLE = False
    # Create dummy function if TMalign not available
    def max_tm_across_refs(*args, **kwargs):
        logger.warning("TMalign not available. Install TMalign and add to PATH.")
        return np.nan, None

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile


def compute_structural_diversity(
    pdb_paths: List[Union[str, Path]],
    atom_type: str = "CA",
    n_threads: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute structural diversity metrics for a set of structures.
    
    Args:
        pdb_paths: List of PDB file paths
        atom_type: Type of atoms to compare
        n_threads: Number of parallel threads
    
    Returns:
        Dictionary with diversity metrics
    """
    if len(pdb_paths) < 2:
        logger.warning("Need at least 2 structures to compute diversity")
        return {}
    
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # Extract coordinates
    coords_list = []
    valid_paths = []
    
    for pdb_path in pdb_paths:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            continue
        
        try:
            struct = PDBFile.read(str(pdb_path)).get_structure()[0]
            
            if atom_type == "CA":
                struct = struct[struct.atom_name == "CA"]
            elif atom_type == "backbone":
                struct = struct[struc.filter_backbone(struct)]
            
            coords_list.append(struct.coord)
            valid_paths.append(pdb_path)
        except Exception as e:
            logger.warning(f"Error reading {pdb_path}: {e}")
            continue
    
    if len(coords_list) < 2:
        return {}
    
    # Normalize lengths (use minimum length)
    min_len = min(len(c) for c in coords_list)
    coords_list = [c[:min_len] for c in coords_list]
    
    n = len(coords_list)
    pairwise_rmsds = []
    
    # Compute pairwise RMSD (use sequential for now to avoid pickle issues)
    # For large datasets, this is still fast enough
    for i in range(n):
        for j in range(i + 1, n):
            try:
                rmsd = compute_rmsd(coords_list[i], coords_list[j], align=True)
                pairwise_rmsds.append(rmsd)
            except:
                pairwise_rmsds.append(np.nan)
    
    pairwise_rmsds = np.array(pairwise_rmsds)
    pairwise_rmsds = pairwise_rmsds[~np.isnan(pairwise_rmsds)]
    
    if len(pairwise_rmsds) == 0:
        return {}
    
    # Compute diversity metrics
    results = {
        "n_structures": n,
        "mean_pairwise_rmsd": float(np.mean(pairwise_rmsds)),
        "median_pairwise_rmsd": float(np.median(pairwise_rmsds)),
        "std_pairwise_rmsd": float(np.std(pairwise_rmsds)),
        "min_pairwise_rmsd": float(np.min(pairwise_rmsds)),
        "max_pairwise_rmsd": float(np.max(pairwise_rmsds)),
        "diversity_score": float(np.mean(pairwise_rmsds)),  # Mean RMSD as diversity score
    }
    
    return results


def compute_structural_diversity_tm_score(
    pdb_paths: List[Union[str, Path]],
    fast: bool = True,
    n_threads: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute structural diversity using TM-scores.
    
    Args:
        pdb_paths: List of PDB file paths
        fast: Use fast mode for TMalign
        n_threads: Number of parallel threads
    
    Returns:
        Dictionary with diversity metrics
    """
    if len(pdb_paths) < 2:
        return {}
    
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # Compute pairwise TM-scores
    # Note: Using sequential computation to avoid pickle issues with nested functions
    # For reasonable dataset sizes (< 50 structures), this is still acceptable
    n = len(pdb_paths)
    pairwise_tms = []
    
    for i in range(n):
        for j in range(i + 1, n):
            try:
                tm = compute_tm_score(pdb_paths[i], pdb_paths[j], fast=fast)
                pairwise_tms.append(tm)
            except:
                pairwise_tms.append(np.nan)
    
    pairwise_tms = np.array(pairwise_tms)
    pairwise_tms = pairwise_tms[~np.isnan(pairwise_tms)]
    
    if len(pairwise_tms) == 0:
        return {}
    
    # Lower TM-score means more diverse
    results = {
        "n_structures": n,
        "mean_pairwise_tm": float(np.mean(pairwise_tms)),
        "median_pairwise_tm": float(np.median(pairwise_tms)),
        "std_pairwise_tm": float(np.std(pairwise_tms)),
        "min_pairwise_tm": float(np.min(pairwise_tms)),
        "max_pairwise_tm": float(np.max(pairwise_tms)),
        "diversity_score_tm": float(1.0 - np.mean(pairwise_tms)),  # Invert for diversity
    }
    
    return results


def compute_novelty_vs_database(
    query_pdbs: List[Union[str, Path]],
    database_pdbs: List[Union[str, Path]],
    fast: bool = True,
    n_threads: Optional[int] = None,
    max_comparisons: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute novelty by comparing generated structures to a reference database.
    
    Args:
        query_pdbs: List of generated PDB file paths
        database_pdbs: List of reference database PDB file paths
        fast: Use fast mode for TMalign
        n_threads: Number of parallel threads
        max_comparisons: Maximum number of database structures to compare against per query
    
    Returns:
        Dictionary with novelty metrics
    """
    if len(query_pdbs) == 0 or len(database_pdbs) == 0:
        return {}
    
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # Limit database size if specified
    if max_comparisons is not None and len(database_pdbs) > max_comparisons:
        # Randomly sample database
        import random
        database_pdbs = random.sample(database_pdbs, max_comparisons)
    
    # For each query, find maximum TM-score against database
    def find_max_tm(query_pdb):
        try:
            max_tm, best_match = max_tm_across_refs(
                str(query_pdb),
                [str(db) for db in database_pdbs],
                n_threads=n_threads,
                fast=fast
            )
            return max_tm, best_match
        except Exception as e:
            logger.warning(f"Error computing novelty for {query_pdb}: {e}")
            return np.nan, None
    
    max_tms = []
    best_matches = []
    
    for query_pdb in query_pdbs:
        max_tm, best_match = find_max_tm(query_pdb)
        max_tms.append(max_tm)
        best_matches.append(best_match)
    
    max_tms = np.array(max_tms)
    max_tms = max_tms[~np.isnan(max_tms)]
    
    if len(max_tms) == 0:
        return {}
    
    # Novelty metrics
    # Lower max TM-score means more novel
    # Structures with TM-score < 0.5 are considered novel
    novel_threshold = 0.5
    
    results = {
        "n_queries": len(query_pdbs),
        "n_database": len(database_pdbs),
        "mean_max_tm": float(np.mean(max_tms)),
        "median_max_tm": float(np.median(max_tms)),
        "std_max_tm": float(np.std(max_tms)),
        "min_max_tm": float(np.min(max_tms)),
        "max_max_tm": float(np.max(max_tms)),
        "novel_fraction": float(np.sum(max_tms < novel_threshold) / len(max_tms)),
        "novel_threshold": novel_threshold,
        "novelty_score": float(1.0 - np.mean(max_tms)),  # Invert for novelty
    }
    
    return results


def compute_sequence_diversity_metrics(
    sequences: List[str]
) -> Dict[str, float]:
    """
    Compute sequence diversity metrics.
    
    Args:
        sequences: List of sequences
    
    Returns:
        Dictionary with sequence diversity metrics
    """
    if len(sequences) < 2:
        return {}
    
    # Use existing function
    diversity = compute_sequence_diversity(sequences)
    
    return diversity


def evaluate_novelty_diversity(
    generated_pdbs: List[Union[str, Path]],
    database_pdbs: Optional[List[Union[str, Path]]] = None,
    sequences: Optional[List[str]] = None,
    atom_type: str = "CA",
    fast: bool = True,
    n_threads: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive novelty and diversity evaluation.
    
    Args:
        generated_pdbs: List of generated PDB file paths
        database_pdbs: Optional list of reference database PDB paths for novelty
        sequences: Optional list of sequences for sequence diversity
        atom_type: Type of atoms to compare
        fast: Use fast mode for TMalign
        n_threads: Number of parallel threads
    
    Returns:
        Dictionary with all novelty and diversity metrics
    """
    results = {}
    
    # Structural diversity
    struct_diversity = compute_structural_diversity(
        generated_pdbs, atom_type=atom_type, n_threads=n_threads
    )
    results.update(struct_diversity)
    
    # Structural diversity using TM-score
    struct_diversity_tm = compute_structural_diversity_tm_score(
        generated_pdbs, fast=fast, n_threads=n_threads
    )
    results.update(struct_diversity_tm)
    
    # Novelty vs database
    if database_pdbs is not None:
        novelty = compute_novelty_vs_database(
            generated_pdbs, database_pdbs, fast=fast, n_threads=n_threads
        )
        results.update(novelty)
    
    # Sequence diversity
    if sequences is not None:
        seq_diversity = compute_sequence_diversity_metrics(sequences)
        results.update({f"seq_{k}": v for k, v in seq_diversity.items()})
    else:
        # Try to extract sequences from PDBs
        try:
            extracted_sequences = []
            for pdb in generated_pdbs:
                seq = extract_sequence_from_pdb(pdb)
                if seq:
                    extracted_sequences.append(seq)
            
            if len(extracted_sequences) >= 2:
                seq_diversity = compute_sequence_diversity_metrics(extracted_sequences)
                results.update({f"seq_{k}": v for k, v in seq_diversity.items()})
        except Exception as e:
            logger.warning(f"Could not extract sequences for diversity: {e}")
    
    return results


def batch_evaluate_novelty_diversity(
    generated_pdbs: List[Union[str, Path]],
    database_pdbs: Optional[List[Union[str, Path]]] = None,
    sequences: Optional[List[str]] = None,
    atom_type: str = "CA",
    fast: bool = True,
    n_threads: Optional[int] = None
) -> pd.DataFrame:
    """
    Batch evaluation of novelty and diversity.
    
    Returns a DataFrame with metrics for the entire set.
    """
    metrics = evaluate_novelty_diversity(
        generated_pdbs,
        database_pdbs=database_pdbs,
        sequences=sequences,
        atom_type=atom_type,
        fast=fast,
        n_threads=n_threads
    )
    
    # Convert to DataFrame (single row)
    df = pd.DataFrame([metrics])
    return df

