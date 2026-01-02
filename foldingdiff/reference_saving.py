"""
Utilities for automatically saving reference structures during sampling.

This module provides functionality to:
1. Load reference structures from CATH dataset
2. Extract motif regions from references
3. Save reference structures alongside generated samples
4. Enable automatic evaluation against references
"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch

from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset
from foldingdiff import angles_and_coords

logger = logging.getLogger(__name__)


def load_reference_structure_from_dataset(
    dataset: CathCanonicalAnglesOnlyDataset,
    index: Optional[int] = None,
    length: Optional[int] = None,
    motif_regions: Optional[List[Tuple[int, int]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a reference structure from the CATH dataset.
    
    Args:
        dataset: CATH dataset to load from
        index: Specific index to load (None = random)
        length: Desired length (will find closest match)
        motif_regions: Motif regions to extract (for motif scaffolding)
    
    Returns:
        Dictionary with:
        - 'pdb_path': Path to original PDB file
        - 'angles': Tensor of angles [length, 6]
        - 'length': Actual length
        - 'motif_regions': Motif regions if provided
        - 'structure_id': CATH structure ID
    """
    try:
        # Get random index if not specified
        if index is None:
            index = np.random.randint(0, len(dataset))
        
        # Get structure from dataset
        item = dataset.__getitem__(index, ignore_zero_center=False)
        
        # Get actual length
        actual_length = item['lengths'].item()
        
        # Filter by length if specified
        if length is not None:
            # Find structures with similar length
            max_attempts = 100
            for _ in range(max_attempts):
                if abs(actual_length - length) <= 10:  # Within 10 residues
                    break
                index = np.random.randint(0, len(dataset))
                item = dataset.__getitem__(index, ignore_zero_center=False)
                actual_length = item['lengths'].item()
        
        # Get PDB file path
        pdb_path = Path(dataset.fnames[index])
        
        # Get angles (add means back)
        angles = item['angles'][:actual_length].clone()
        
        # Add means back (model learned mean-centered)
        angles[:, 2] += np.pi  # omega: add 180° for trans
        angles[:, 3] += 1.92  # tau: add ~110°
        angles[:, 4] += 2.01  # CA:C:1N: add ~115°
        angles[:, 5] += 2.11  # C:1N:1CA: add ~121°
        
        # Extract structure ID
        structure_id = pdb_path.stem
        
        result = {
            'pdb_path': pdb_path,
            'angles': angles,
            'length': actual_length,
            'structure_id': structure_id,
        }
        
        # Add motif regions if provided
        if motif_regions:
            result['motif_regions'] = motif_regions
        
        return result
    
    except Exception as e:
        logger.error(f"Error loading reference structure: {e}")
        return None


def save_reference_structure(
    reference_data: Dict[str, Any],
    output_dir: Path,
    sample_id: int,
    save_pdb: bool = True
) -> Optional[Path]:
    """
    Save reference structure to output directory.
    
    Args:
        reference_data: Dictionary from load_reference_structure_from_dataset
        output_dir: Output directory for samples
        sample_id: Sample ID (for naming)
        save_pdb: Whether to save PDB file
    
    Returns:
        Path to saved reference PDB file, or None if failed
    """
    try:
        # Create references directory
        ref_dir = output_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        # Copy original PDB file
        original_pdb = reference_data['pdb_path']
        ref_pdb = ref_dir / f"reference_{sample_id:04d}.pdb"
        
        if original_pdb.exists():
            # Copy original PDB
            shutil.copy(original_pdb, ref_pdb)
            logger.debug(f"Saved reference PDB: {ref_pdb}")
        else:
            # Reconstruct from angles if original not found
            if save_pdb:
                angles = reference_data['angles']
                
                # Create dataframe with angles
                import pandas as pd
                angles_df = pd.DataFrame({
                    'phi': angles[:, 0].numpy(),
                    'psi': angles[:, 1].numpy(),
                    'omega': angles[:, 2].numpy(),
                    'tau': angles[:, 3].numpy(),
                    'CA:C:1N': angles[:, 4].numpy(),
                    'C:1N:1CA': angles[:, 5].numpy(),
                    # Standard peptide bond distances
                    '0C:1N': 1.329,
                    'N:CA': 1.458,
                    'CA:C': 1.525,
                })
                
                # Build 3D structure with NERF
                angles_and_coords.create_new_chain_nerf(
                    str(ref_pdb),
                    angles_df,
                    angles_to_set=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
                    dists_to_set=['0C:1N', 'N:CA', 'CA:C'],
                )
                logger.debug(f"Reconstructed reference PDB from angles: {ref_pdb}")
        
        return ref_pdb
    
    except Exception as e:
        logger.error(f"Error saving reference structure: {e}")
        return None


def load_and_save_references(
    n_samples: int,
    output_dir: Path,
    dataset: Optional[CathCanonicalAnglesOnlyDataset] = None,
    length: Optional[int] = None,
    motif_regions: Optional[List[Tuple[int, int]]] = None,
    save_pdb: bool = True,
    seed: Optional[int] = None
) -> List[Optional[Path]]:
    """
    Load and save multiple reference structures from CATH dataset.
    
    Args:
        n_samples: Number of reference structures to load
        output_dir: Output directory for samples
        dataset: CATH dataset (will create if None)
        length: Desired length (None = any length)
        motif_regions: Motif regions (for motif scaffolding)
        save_pdb: Whether to save PDB files
        seed: Random seed for reproducibility
    
    Returns:
        List of paths to saved reference PDB files
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create dataset if not provided
    if dataset is None:
        try:
            dataset = CathCanonicalAnglesOnlyDataset(
                pdbs="cath",
                split=None,  # Use all data
                pad=512,
                min_length=40 if length is None else max(40, length - 20),
            )
            logger.info(f"Created CATH dataset with {len(dataset)} structures")
        except Exception as e:
            logger.error(f"Failed to create CATH dataset: {e}")
            return [None] * n_samples
    
    saved_paths = []
    
    for i in range(n_samples):
        # Load reference structure
        ref_data = load_reference_structure_from_dataset(
            dataset=dataset,
            index=None,  # Random
            length=length,
            motif_regions=motif_regions
        )
        
        if ref_data is None:
            logger.warning(f"Failed to load reference structure {i}")
            saved_paths.append(None)
            continue
        
        # Save reference structure
        ref_path = save_reference_structure(
            reference_data=ref_data,
            output_dir=output_dir,
            sample_id=i,
            save_pdb=save_pdb
        )
        
        saved_paths.append(ref_path)
        
        if ref_path:
            logger.debug(f"Saved reference {i+1}/{n_samples}: {ref_path.name}")
    
    n_saved = sum(1 for p in saved_paths if p is not None)
    logger.info(f"Saved {n_saved}/{n_samples} reference structures to {output_dir / 'references'}")
    
    return saved_paths

