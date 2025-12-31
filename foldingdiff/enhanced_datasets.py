"""
Enhanced datasets that provide rich structural features.

Extends existing datasets to include:
- Backbone coordinates
- Local frames
- Pairwise distances
- Secondary structure predictions
"""
import logging
from typing import *

import torch
import numpy as np

from foldingdiff.datasets import (
    CathCanonicalAnglesOnlyDataset,
    NoisedAnglesDataset,
)
from foldingdiff.motif_scaffolding import MotifScaffoldingDataset
from foldingdiff.embeddings import (
    angles_to_coords_simple,
    compute_secondary_structure_simple,
)


class EnhancedCathDataset(CathCanonicalAnglesOnlyDataset):
    """
    CATH dataset with enhanced features.
    
    Provides:
    - Backbone angles (existing)
    - Backbone coordinates (N, CA, C, O)
    - Amino acid types (if available)
    - Secondary structure predictions
    
    Args:
        compute_coords: Whether to compute coordinates from angles
        compute_ss: Whether to compute secondary structure
        **kwargs: Arguments passed to CathCanonicalAnglesOnlyDataset
    """
    
    def __init__(
        self,
        compute_coords: bool = True,
        compute_ss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compute_coords = compute_coords
        self.compute_ss = compute_ss
        
        logging.info(
            f"EnhancedCathDataset: coords={compute_coords}, ss={compute_ss}"
        )
    
    def __getitem__(self, index, ignore_zero_center: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get item with enhanced features.
        
        Returns dictionary with:
        - angles: [seq_len, 6] backbone angles
        - coords: [seq_len, 4, 3] backbone coordinates (if compute_coords)
        - aa_types: [seq_len] amino acid types (20 = unknown)
        - secondary_structure: [seq_len, 3] SS predictions (if compute_ss)
        - attn_mask: [seq_len] attention mask
        - position_ids: [seq_len] position IDs
        - lengths: scalar sequence length
        """
        # Get base item
        item = super().__getitem__(index, ignore_zero_center)
        
        # Compute coordinates from angles
        if self.compute_coords:
            # Use simplified coordinate computation
            # For production, should use full NERF algorithm
            angles_batch = item['angles'].unsqueeze(0)  # Add batch dim
            coords = angles_to_coords_simple(angles_batch)
            item['coords_computed'] = coords.squeeze(0)  # Remove batch dim
        
        # Compute secondary structure
        if self.compute_ss:
            angles_batch = item['angles'].unsqueeze(0)
            ss = compute_secondary_structure_simple(angles_batch)
            item['secondary_structure'] = ss.squeeze(0)
        
        # Add amino acid types (unknown for now)
        # In production, extract from PDB files
        item['aa_types'] = torch.full(
            (item['angles'].shape[0],),
            20,  # 20 = unknown
            dtype=torch.long
        )
        
        return item


class EnhancedMotifScaffoldingDataset(MotifScaffoldingDataset):
    """
    Motif scaffolding dataset with enhanced features.
    
    Wraps EnhancedCathDataset to provide both motif conditioning
    and rich structural features.
    
    Args:
        base_dataset: EnhancedCathDataset instance
        **kwargs: Arguments passed to MotifScaffoldingDataset
    """
    
    def __init__(
        self,
        base_dataset: EnhancedCathDataset,
        **kwargs
    ):
        # Initialize parent with enhanced dataset
        super().__init__(base_dataset, **kwargs)
        
        logging.info("EnhancedMotifScaffoldingDataset initialized")
    
    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Get item with motif conditioning and enhanced features.
        
        Returns all features from EnhancedCathDataset plus:
        - motif_mask: [seq_len, 1] motif mask
        - motif_angles: [seq_len, n_features] clean motif angles
        - num_motifs: number of motif regions
        """
        # Get base item with motif conditioning
        item = super().__getitem__(index, use_t_val, ignore_zero_center)
        
        # Enhanced features are already included from base dataset
        # Just ensure they're present
        if 'coords_computed' not in item and hasattr(self.dset, 'compute_coords'):
            if self.dset.compute_coords:
                angles_batch = item['angles'].unsqueeze(0)
                coords = angles_to_coords_simple(angles_batch)
                item['coords_computed'] = coords.squeeze(0)
        
        if 'secondary_structure' not in item and hasattr(self.dset, 'compute_ss'):
            if self.dset.compute_ss:
                angles_batch = item['angles'].unsqueeze(0)
                ss = compute_secondary_structure_simple(angles_batch)
                item['secondary_structure'] = ss.squeeze(0)
        
        return item


def create_enhanced_dataset(
    split: Optional[str] = "train",
    pad: int = 128,
    min_length: int = 40,
    toy: int = 0,
    compute_coords: bool = True,
    compute_ss: bool = True,
    use_motif_scaffolding: bool = False,
    motif_length_range: Tuple[int, int] = (5, 20),
    motif_prob: float = 0.8,
    timesteps: int = 1000,
    beta_schedule: str = "cosine",
) -> Union[EnhancedCathDataset, EnhancedMotifScaffoldingDataset]:
    """
    Factory function to create enhanced datasets.
    
    Args:
        split: Dataset split ('train', 'validation', 'test', or None)
        pad: Padding length
        min_length: Minimum sequence length
        toy: Toy dataset size (0 = full dataset)
        compute_coords: Compute coordinates from angles
        compute_ss: Compute secondary structure
        use_motif_scaffolding: Wrap with motif scaffolding
        motif_length_range: Range of motif lengths
        motif_prob: Probability of including motif
        timesteps: Number of diffusion/flow timesteps
        beta_schedule: Noise schedule
    
    Returns:
        dataset: Enhanced dataset (with or without motif scaffolding)
    """
    # Create base enhanced dataset
    base_dataset = EnhancedCathDataset(
        pdbs="cath",
        split=split,
        pad=pad,
        min_length=min_length,
        toy=toy,
        compute_coords=compute_coords,
        compute_ss=compute_ss,
    )
    
    # Optionally wrap with motif scaffolding
    if use_motif_scaffolding:
        dataset = EnhancedMotifScaffoldingDataset(
            base_dataset=base_dataset,
            motif_length_range=motif_length_range,
            motif_prob=motif_prob,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
        )
    else:
        # Wrap with noising for training
        dataset = NoisedAnglesDataset(
            dset=base_dataset,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
        )
    
    logging.info(f"Created enhanced dataset: {dataset}")
    
    return dataset


class CoordinateAugmentation:
    """
    Data augmentation for protein coordinates.
    
    Applies random rotations and translations to coordinates
    while keeping angles unchanged (for data augmentation).
    
    Args:
        rotation_std: Standard deviation for random rotations (radians)
        translation_std: Standard deviation for random translations (Angstroms)
    """
    
    def __init__(
        self,
        rotation_std: float = 0.1,
        translation_std: float = 1.0,
    ):
        self.rotation_std = rotation_std
        self.translation_std = translation_std
    
    def __call__(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply random rotation and translation.
        
        Args:
            coords: [seq_len, n_atoms, 3] or [batch, seq_len, n_atoms, 3]
        
        Returns:
            coords_aug: Augmented coordinates (same shape)
        """
        # Random rotation matrix
        angles = torch.randn(3) * self.rotation_std
        R = self._rotation_matrix(angles)
        
        # Random translation
        t = torch.randn(3) * self.translation_std
        
        # Apply transformation
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        coords_aug = torch.matmul(coords_flat, R.T) + t
        coords_aug = coords_aug.reshape(original_shape)
        
        return coords_aug
    
    def _rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Create 3D rotation matrix from Euler angles.
        
        Args:
            angles: [3] rotation angles (rx, ry, rz)
        
        Returns:
            R: [3, 3] rotation matrix
        """
        rx, ry, rz = angles
        
        # Rotation around x-axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rx), -torch.sin(rx)],
            [0, torch.sin(rx), torch.cos(rx)]
        ])
        
        # Rotation around y-axis
        Ry = torch.tensor([
            [torch.cos(ry), 0, torch.sin(ry)],
            [0, 1, 0],
            [-torch.sin(ry), 0, torch.cos(ry)]
        ])
        
        # Rotation around z-axis
        Rz = torch.tensor([
            [torch.cos(rz), -torch.sin(rz), 0],
            [torch.sin(rz), torch.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        
        return R
