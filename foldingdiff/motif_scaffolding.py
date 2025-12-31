"""
Motif scaffolding dataset and utilities for conditional protein generation.

This module provides functionality for training and sampling conditional
protein generation models that preserve specific structural motifs while
generating scaffold regions.
"""
import logging
from typing import *

import numpy as np
import torch
from torch.utils.data import Dataset

from foldingdiff.datasets import NoisedAnglesDataset


class MotifScaffoldingDataset(NoisedAnglesDataset):
    """
    Dataset for motif scaffolding training.
    
    Randomly selects contiguous regions as "motifs" during training.
    The model learns to generate scaffold regions while preserving motifs.
    
    Args:
        dset: Base dataset (e.g., CathCanonicalAnglesOnlyDataset)
        motif_length_range: Tuple of (min_length, max_length) for motif regions
        motif_prob: Probability of including a motif (vs unconditional generation)
        max_motifs: Maximum number of motif regions per structure
        min_scaffold_length: Minimum length of scaffold regions between motifs
        **kwargs: Additional arguments passed to NoisedAnglesDataset
    """
    
    def __init__(
        self,
        dset: Dataset,
        motif_length_range: Tuple[int, int] = (5, 20),
        motif_prob: float = 0.8,
        max_motifs: int = 1,
        min_scaffold_length: int = 10,
        **kwargs
    ):
        super().__init__(dset, **kwargs)
        
        self.motif_length_range = motif_length_range
        self.motif_prob = motif_prob
        self.max_motifs = max_motifs
        self.min_scaffold_length = min_scaffold_length
        
        # Random number generator for motif selection
        self.motif_rng = np.random.default_rng(seed=42)
        
        logging.info(
            f"MotifScaffoldingDataset: motif_length={motif_length_range}, "
            f"prob={motif_prob}, max_motifs={max_motifs}"
        )
    
    def _sample_motif_regions(
        self, 
        seq_length: int
    ) -> List[Tuple[int, int]]:
        """
        Sample motif regions for a sequence.
        
        Args:
            seq_length: Length of the sequence
            
        Returns:
            List of (start, end) tuples for motif regions
        """
        # Decide whether to include motifs
        if self.motif_rng.random() > self.motif_prob:
            return []  # Unconditional generation
        
        # Sample number of motifs
        num_motifs = self.motif_rng.integers(1, self.max_motifs + 1)
        
        motif_regions = []
        available_length = seq_length
        
        for _ in range(num_motifs):
            # Check if we have enough space for another motif
            min_required = (
                self.motif_length_range[0] + 
                self.min_scaffold_length * (len(motif_regions) + 1)
            )
            if available_length < min_required:
                break
            
            # Sample motif length
            max_len = min(
                self.motif_length_range[1],
                available_length - self.min_scaffold_length * (len(motif_regions) + 1)
            )
            if max_len < self.motif_length_range[0]:
                break
                
            motif_len = self.motif_rng.integers(
                self.motif_length_range[0], 
                max_len + 1
            )
            
            # Sample motif position
            # Ensure minimum scaffold length before and after
            if len(motif_regions) == 0:
                # First motif
                max_start = seq_length - motif_len - self.min_scaffold_length
                if max_start < self.min_scaffold_length:
                    break
                motif_start = self.motif_rng.integers(
                    self.min_scaffold_length,
                    max_start + 1
                )
            else:
                # Subsequent motifs - place after previous motif
                prev_end = motif_regions[-1][1]
                max_start = seq_length - motif_len - self.min_scaffold_length
                if max_start < prev_end + self.min_scaffold_length:
                    break
                motif_start = self.motif_rng.integers(
                    prev_end + self.min_scaffold_length,
                    max_start + 1
                )
            
            motif_end = motif_start + motif_len
            motif_regions.append((motif_start, motif_end))
            available_length -= motif_len
        
        return motif_regions
    
    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
        stochastic_centering: bool = True,
        centering_std: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Get item with motif conditioning.
        
        Args:
            stochastic_centering: If True, add small random noise to motif angles
                to prevent model from learning fixed offsets (improves generalization)
            centering_std: Standard deviation for stochastic centering noise
        
        Returns dictionary with additional keys:
            - motif_mask: [seq_len, 1] binary mask (1=motif, 0=scaffold)
            - motif_angles: [seq_len, n_features] clean angles for motif regions
            - num_motifs: number of motif regions
        """
        # Get base noised item
        item = super().__getitem__(
            index, 
            use_t_val=use_t_val,
            ignore_zero_center=ignore_zero_center
        )
        
        # Get sequence length
        seq_length = item['lengths'].item()
        
        # Sample motif regions
        motif_regions = self._sample_motif_regions(seq_length)
        
        # Create motif mask
        motif_mask = torch.zeros(self.pad, 1, dtype=torch.float32)
        for start, end in motif_regions:
            motif_mask[start:end, 0] = 1.0
        
        # Store clean angles for motif regions
        # These will be used during sampling to preserve motif geometry
        motif_angles = torch.zeros_like(item['corrupted'])
        if len(motif_regions) > 0:
            # Get clean angles from the base dataset item
            clean_angles = item[self.dset_key]
            motif_angles = clean_angles.clone()
            
            # Apply stochastic centering: add small random noise to motif angles
            # This prevents the model from learning fixed offsets between scaffold and motif
            # and improves generalization during inference
            if stochastic_centering and centering_std > 0:
                # Only add noise to motif regions
                mask_expanded = motif_mask.expand_as(motif_angles)
                noise = torch.randn_like(motif_angles) * centering_std
                # For angular features, wrap the noise appropriately
                # For now, just add small noise (centering_std should be small)
                motif_angles = motif_angles + mask_expanded * noise
        
        # Add to return dictionary
        item['motif_mask'] = motif_mask
        item['motif_angles'] = motif_angles
        item['num_motifs'] = torch.tensor(len(motif_regions), dtype=torch.long)
        # Note: motif_regions is not included as it can't be collated
        # Store as metadata if needed for debugging
        
        return item
    
    def __str__(self) -> str:
        return (
            f"MotifScaffoldingDataset wrapping {self.dset} with "
            f"{len(self)} examples, motif_length={self.motif_length_range}, "
            f"prob={self.motif_prob}, schedule={self.schedule}-{self.timesteps}"
        )


def create_motif_mask_from_regions(
    regions: List[Tuple[int, int]],
    seq_length: int,
    pad_length: int
) -> torch.Tensor:
    """
    Create a motif mask from a list of motif regions.
    
    Args:
        regions: List of (start, end) tuples for motif regions
        seq_length: Actual sequence length
        pad_length: Padded sequence length
        
    Returns:
        motif_mask: [pad_length, 1] binary mask
    """
    motif_mask = torch.zeros(pad_length, 1, dtype=torch.float32)
    for start, end in regions:
        if end <= seq_length:
            motif_mask[start:end, 0] = 1.0
    return motif_mask


def extract_motif_coords(
    coords: torch.Tensor,
    motif_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract coordinates for motif regions.
    
    Args:
        coords: [seq_len, 3] or [seq_len, n_atoms, 3] coordinates
        motif_mask: [seq_len, 1] binary mask
        
    Returns:
        motif_coords: Coordinates for motif regions only
    """
    mask_idx = motif_mask.squeeze(-1) > 0.5
    return coords[mask_idx]


def compute_motif_rmsd(
    generated_coords: torch.Tensor,
    target_coords: torch.Tensor,
    motif_mask: torch.Tensor
) -> float:
    """
    Compute RMSD between generated and target coordinates for motif regions.
    
    Args:
        generated_coords: [seq_len, 3] generated CA coordinates
        target_coords: [seq_len, 3] target CA coordinates
        motif_mask: [seq_len, 1] binary mask
        
    Returns:
        rmsd: Root mean square deviation in Angstroms
    """
    mask_idx = motif_mask.squeeze(-1) > 0.5
    
    gen_motif = generated_coords[mask_idx]
    tgt_motif = target_coords[mask_idx]
    
    if len(gen_motif) == 0:
        return 0.0
    
    # Compute RMSD
    diff = gen_motif - tgt_motif
    rmsd = torch.sqrt((diff ** 2).sum(-1).mean()).item()
    
    return rmsd
