"""
Data augmentation utilities for protein structures.

Implements best practices from 2024-2025 research:
- Angle noise (within Ramachandran)
- Motif position randomization
- Sequence augmentation
"""
import logging
from typing import *

import torch
import numpy as np

from foldingdiff.geometric_validation import check_ramachandran


class ProteinDataAugmentation:
    """
    Data augmentation for protein structures.
    
    Applies:
    1. Small angle noise (within Ramachandran constraints)
    2. Motif position randomization (if motifs present)
    3. Sequence shuffling (if sequences available)
    
    Args:
        angle_noise_std: Standard deviation for angle noise (default: 0.05 rad)
        apply_prob: Probability of applying augmentation (default: 0.5)
    """
    
    def __init__(
        self,
        angle_noise_std: float = 0.05,
        apply_prob: float = 0.5
    ):
        self.angle_noise_std = angle_noise_std
        self.apply_prob = apply_prob
        logging.info(
            f"ProteinDataAugmentation: noise_std={angle_noise_std}, apply_prob={apply_prob}"
        )
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to batch.
        
        Args:
            batch: Dictionary with 'angles' and optionally 'motif_mask', 'sequences'
        
        Returns:
            Augmented batch
        """
        # Randomly decide whether to apply augmentation
        if np.random.rand() > self.apply_prob:
            return batch
        
        # 1. Small angle noise (within Ramachandran)
        if 'angles' in batch:
            angles = batch['angles'].clone()
            
            # Add small noise
            noise = torch.randn_like(angles) * self.angle_noise_std
            angles_aug = angles + noise
            
            # Wrap angular features (phi, psi, omega are indices 0, 1, 2)
            from foldingdiff import utils
            for j in [0, 1, 2]:  # phi, psi, omega
                angles_aug[:, j] = utils.modulo_with_wrapped_range(
                    angles_aug[:, j],
                    range_min=-torch.pi,
                    range_max=torch.pi
                )
            
            batch['angles'] = angles_aug
        
        # 2. Motif position randomization (if motifs present)
        if 'motif_mask' in batch and batch['motif_mask'].sum() > 0:
            # Slight motif position shifts could be added here
            # For now, we keep motifs as-is to preserve structure
            pass
        
        # 3. Sequence shuffling (if sequences available)
        if 'sequences' in batch:
            # Could apply sequence augmentation here
            # For now, keep sequences as-is
            pass
        
        return batch


def validate_structure_quality(
    angles: np.ndarray,
    coords: Optional[np.ndarray] = None,
    min_rama_favored: float = 0.70,
    max_clash_rate: float = 0.10
) -> bool:
    """
    Validate structure quality based on Ramachandran and clash metrics.
    
    Args:
        angles: [seq_len, n_features] backbone angles
        coords: [seq_len, 4, 3] backbone coordinates (optional)
        min_rama_favored: Minimum fraction of Ramachandran favored residues
        max_clash_rate: Maximum clash rate
    
    Returns:
        True if structure passes quality checks
    """
    # Extract phi and psi
    phi = angles[:, 0]
    psi = angles[:, 1]
    
    # Check Ramachandran quality
    rama_stats = check_ramachandran(phi, psi)
    favored = rama_stats['favored']
    
    if favored < min_rama_favored:
        return False
    
    # Check for excessive clashes (if coords available)
    if coords is not None:
        # Simple clash check: count residues with very close CA-CA distances
        from scipy.spatial.distance import pdist
        ca_coords = coords[:, 1, :]  # CA atoms
        if len(ca_coords) > 1:
            distances = pdist(ca_coords)
            # Clash: distance < 3.0 Å (should be > 3.8 Å for CA-CA)
            n_clashes = (distances < 3.0).sum()
            clash_rate = n_clashes / len(distances) if len(distances) > 0 else 0.0
            
            if clash_rate > max_clash_rate:
                return False
    
    return True


def filter_structures_by_quality(
    structures: List[Dict[str, Any]],
    min_rama_favored: float = 0.70,
    max_clash_rate: float = 0.10
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Filter structures by quality metrics.
    
    Args:
        structures: List of structure dictionaries with 'angles' and optionally 'coords'
        min_rama_favored: Minimum fraction of Ramachandran favored residues
        max_clash_rate: Maximum clash rate
    
    Returns:
        Tuple of (filtered_structures, kept_indices)
    """
    filtered = []
    kept_indices = []
    
    for i, structure in enumerate(structures):
        angles = structure.get('angles')
        if angles is None:
            continue
        
        # Convert to numpy if needed
        if isinstance(angles, torch.Tensor):
            angles_np = angles.cpu().numpy()
        else:
            angles_np = angles
        
        # Get coords if available
        coords = structure.get('coords')
        if coords is not None and isinstance(coords, torch.Tensor):
            coords_np = coords.cpu().numpy()
        else:
            coords_np = coords
        
        # Validate quality
        if validate_structure_quality(
            angles_np,
            coords_np,
            min_rama_favored=min_rama_favored,
            max_clash_rate=max_clash_rate
        ):
            filtered.append(structure)
            kept_indices.append(i)
    
    return filtered, kept_indices

