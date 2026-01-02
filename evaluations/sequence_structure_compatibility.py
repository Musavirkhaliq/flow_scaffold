"""
Sequence-structure compatibility metrics.

Evaluates how well generated sequences match their structures and vice versa.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Amino acid properties for compatibility analysis
AA_PROPERTIES = {
    'A': {'hydrophobic': True, 'small': True, 'polar': False},
    'R': {'hydrophobic': False, 'small': False, 'polar': True},
    'N': {'hydrophobic': False, 'small': True, 'polar': True},
    'D': {'hydrophobic': False, 'small': True, 'polar': True},
    'C': {'hydrophobic': True, 'small': True, 'polar': False},
    'Q': {'hydrophobic': False, 'small': False, 'polar': True},
    'E': {'hydrophobic': False, 'small': False, 'polar': True},
    'G': {'hydrophobic': True, 'small': True, 'polar': False},
    'H': {'hydrophobic': False, 'small': False, 'polar': True},
    'I': {'hydrophobic': True, 'small': False, 'polar': False},
    'L': {'hydrophobic': True, 'small': False, 'polar': False},
    'K': {'hydrophobic': False, 'small': False, 'polar': True},
    'M': {'hydrophobic': True, 'small': False, 'polar': False},
    'F': {'hydrophobic': True, 'small': False, 'polar': False},
    'P': {'hydrophobic': True, 'small': True, 'polar': False},
    'S': {'hydrophobic': False, 'small': True, 'polar': True},
    'T': {'hydrophobic': False, 'small': True, 'polar': True},
    'W': {'hydrophobic': True, 'small': False, 'polar': False},
    'Y': {'hydrophobic': True, 'small': False, 'polar': True},
    'V': {'hydrophobic': True, 'small': True, 'polar': False},
}

# BLOSUM62 substitution matrix (simplified - key substitutions)
BLOSUM62_SIMILAR = {
    'A': ['A', 'S', 'G'],
    'R': ['R', 'K', 'H'],
    'N': ['N', 'D', 'S'],
    'D': ['D', 'E', 'N'],
    'C': ['C', 'S'],
    'Q': ['Q', 'E', 'N'],
    'E': ['E', 'D', 'Q'],
    'G': ['G', 'A', 'S'],
    'H': ['H', 'Y', 'F'],
    'I': ['I', 'L', 'V', 'M'],
    'L': ['L', 'I', 'V', 'M'],
    'K': ['K', 'R', 'Q'],
    'M': ['M', 'I', 'L', 'V'],
    'F': ['F', 'Y', 'W', 'H'],
    'P': ['P'],
    'S': ['S', 'T', 'A', 'N'],
    'T': ['T', 'S', 'N'],
    'W': ['W', 'F', 'Y'],
    'Y': ['Y', 'F', 'W', 'H'],
    'V': ['V', 'I', 'L', 'M'],
}


def extract_sequence_from_pdb(pdb_path: Union[str, Path]) -> Optional[str]:
    """
    Extract amino acid sequence from PDB file.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Sequence string or None if extraction fails
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        return None
    
    try:
        struct = PDBFile.read(str(pdb_path)).get_structure()[0]
        # Get CA atoms to determine sequence
        ca_atoms = struct[struct.atom_name == "CA"]
        
        # Extract residue names
        sequence = ""
        for res_name in ca_atoms.res_name:
            # Convert 3-letter to 1-letter code
            aa_3to1 = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            }
            sequence += aa_3to1.get(res_name, 'X')
        
        return sequence
    except Exception as e:
        logger.error(f"Error extracting sequence from {pdb_path}: {e}")
        return None


def compute_sequence_recovery_rate(
    generated_sequence: str,
    reference_sequence: str
) -> Dict[str, float]:
    """
    Compute sequence recovery rate metrics.
    
    Args:
        generated_sequence: Generated amino acid sequence
        reference_sequence: Reference/target sequence
    
    Returns:
        Dictionary with recovery metrics
    """
    if len(generated_sequence) != len(reference_sequence):
        logger.warning("Sequences have different lengths, truncating to shorter length")
        min_len = min(len(generated_sequence), len(reference_sequence))
        generated_sequence = generated_sequence[:min_len]
        reference_sequence = reference_sequence[:min_len]
    
    if len(generated_sequence) == 0:
        return {}
    
    # Exact matches
    exact_matches = sum(g == r for g, r in zip(generated_sequence, reference_sequence))
    exact_recovery = exact_matches / len(generated_sequence)
    
    # Similar matches (using BLOSUM62-like similarity)
    similar_matches = 0
    for g, r in zip(generated_sequence, reference_sequence):
        if g == r:
            similar_matches += 1
        elif g in BLOSUM62_SIMILAR.get(r, []):
            similar_matches += 1
    
    similar_recovery = similar_matches / len(generated_sequence)
    
    # Property-based matches (hydrophobic, polar, etc.)
    property_matches = 0
    for g, r in zip(generated_sequence, reference_sequence):
        g_props = AA_PROPERTIES.get(g, {})
        r_props = AA_PROPERTIES.get(r, {})
        if g_props and r_props:
            # Match if same hydrophobicity
            if g_props.get('hydrophobic') == r_props.get('hydrophobic'):
                property_matches += 1
    
    property_recovery = property_matches / len(generated_sequence)
    
    results = {
        "exact_recovery_rate": float(exact_recovery),
        "similar_recovery_rate": float(similar_recovery),
        "property_recovery_rate": float(property_recovery),
        "n_exact_matches": exact_matches,
        "n_similar_matches": similar_matches,
        "n_property_matches": property_matches,
        "sequence_length": len(generated_sequence),
    }
    
    return results


def compute_sequence_similarity(
    seq1: str,
    seq2: str
) -> Dict[str, float]:
    """
    Compute sequence similarity metrics.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
    
    Returns:
        Dictionary with similarity metrics
    """
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    if len(seq1) == 0:
        return {}
    
    # Identity
    identity = sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)
    
    # Similarity (using BLOSUM62-like)
    similarity = 0
    for a, b in zip(seq1, seq2):
        if a == b:
            similarity += 1
        elif a in BLOSUM62_SIMILAR.get(b, []):
            similarity += 1
    
    similarity = similarity / len(seq1)
    
    # Hamming distance
    hamming = sum(a != b for a, b in zip(seq1, seq2))
    
    results = {
        "sequence_identity": float(identity),
        "sequence_similarity": float(similarity),
        "hamming_distance": int(hamming),
        "sequence_length": len(seq1),
    }
    
    return results


def compute_sequence_diversity(
    sequences: List[str]
) -> Dict[str, float]:
    """
    Compute diversity metrics for a set of sequences.
    
    Args:
        sequences: List of sequences
    
    Returns:
        Dictionary with diversity metrics
    """
    if len(sequences) == 0:
        return {}
    
    # Ensure all sequences have same length
    min_len = min(len(s) for s in sequences)
    sequences = [s[:min_len] for s in sequences]
    
    # Position-wise diversity (Shannon entropy)
    position_entropies = []
    for pos in range(min_len):
        aa_counts = Counter(s[pos] for s in sequences)
        total = sum(aa_counts.values())
        entropy = -sum((count / total) * np.log2(count / total) for count in aa_counts.values())
        position_entropies.append(entropy)
    
    # Pairwise sequence identity
    pairwise_identities = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            identity = sum(a == b for a, b in zip(sequences[i], sequences[j])) / min_len
            pairwise_identities.append(identity)
    
    results = {
        "n_sequences": len(sequences),
        "sequence_length": min_len,
        "mean_position_entropy": float(np.mean(position_entropies)),
        "std_position_entropy": float(np.std(position_entropies)),
        "mean_pairwise_identity": float(np.mean(pairwise_identities)) if pairwise_identities else 0.0,
        "std_pairwise_identity": float(np.std(pairwise_identities)) if pairwise_identities else 0.0,
        "diversity_score": float(np.mean(position_entropies) * (1 - np.mean(pairwise_identities)) if pairwise_identities else np.mean(position_entropies)),
    }
    
    return results


def evaluate_sequence_structure_compatibility(
    generated_pdb: Union[str, Path],
    reference_pdb: Union[str, Path],
    generated_sequence: Optional[str] = None,
    reference_sequence: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate sequence-structure compatibility.
    
    Args:
        generated_pdb: Path to generated PDB file
        reference_pdb: Path to reference PDB file
        generated_sequence: Optional generated sequence (extracted from PDB if not provided)
        reference_sequence: Optional reference sequence (extracted from PDB if not provided)
    
    Returns:
        Dictionary with compatibility metrics
    """
    results = {}
    
    # Extract sequences if not provided
    if generated_sequence is None:
        generated_sequence = extract_sequence_from_pdb(generated_pdb)
    if reference_sequence is None:
        reference_sequence = extract_sequence_from_pdb(reference_pdb)
    
    if generated_sequence is None or reference_sequence is None:
        logger.warning("Could not extract sequences from PDB files")
        return results
    
    # Sequence recovery
    recovery = compute_sequence_recovery_rate(generated_sequence, reference_sequence)
    results.update(recovery)
    
    # Sequence similarity
    similarity = compute_sequence_similarity(generated_sequence, reference_sequence)
    results.update(similarity)
    
    return results


def batch_evaluate_sequence_structure_compatibility(
    generated_pdbs: List[Union[str, Path]],
    reference_pdbs: List[Union[str, Path]],
    generated_sequences: Optional[List[str]] = None,
    reference_sequences: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Batch evaluation of sequence-structure compatibility.
    
    Args:
        generated_pdbs: List of generated PDB file paths
        reference_pdbs: List of reference PDB file paths
        generated_sequences: Optional list of generated sequences
        reference_sequences: Optional list of reference sequences
    
    Returns:
        DataFrame with evaluation results
    """
    assert len(generated_pdbs) == len(reference_pdbs), "Lists must have same length"
    
    if generated_sequences is not None:
        assert len(generated_sequences) == len(generated_pdbs)
    if reference_sequences is not None:
        assert len(reference_sequences) == len(reference_pdbs)
    
    results = []
    for i, (gen_pdb, ref_pdb) in enumerate(zip(generated_pdbs, reference_pdbs)):
        try:
            gen_seq = generated_sequences[i] if generated_sequences else None
            ref_seq = reference_sequences[i] if reference_sequences else None
            
            metrics = evaluate_sequence_structure_compatibility(
                gen_pdb, ref_pdb, gen_seq, ref_seq
            )
            metrics["generated_pdb"] = str(gen_pdb)
            metrics["reference_pdb"] = str(ref_pdb)
            metrics["pair_index"] = i
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating pair {i}: {e}")
            results.append({
                "generated_pdb": str(gen_pdb),
                "reference_pdb": str(ref_pdb),
                "pair_index": i,
                "error": str(e)
            })
    
    return pd.DataFrame(results)

