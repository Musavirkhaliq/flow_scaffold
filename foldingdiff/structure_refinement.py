"""
Structure refinement utilities for post-processing generated structures.

Provides functions to:
- Apply energy minimization
- Refine geometric quality
- Fix clashes
- Improve Ramachandran quality
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from foldingdiff import geometric_validation, angles_and_coords
import pandas as pd

logger = logging.getLogger(__name__)


def refine_angles_by_ramachandran(
    angles: Union[np.ndarray, torch.Tensor],
    target_favored: float = 0.5,
    max_iterations: int = 10
) -> Union[np.ndarray, torch.Tensor]:
    """
    Refine angles to improve Ramachandran quality.
    
    Moves outliers towards nearest favored/allowed region.
    
    Args:
        angles: Angle tensor [n_residues, 6]
        target_favored: Target fraction in favored regions
        max_iterations: Maximum refinement iterations
    
    Returns:
        Refined angles
    """
    if isinstance(angles, torch.Tensor):
        is_torch = True
        device = angles.device
        angles = angles.cpu().numpy()
    else:
        is_torch = False
    
    if angles.ndim == 1:
        angles = angles.reshape(1, -1)
    
    refined = angles.copy()
    phi = refined[:, 0]
    psi = refined[:, 1]
    
    for iteration in range(max_iterations):
        # Check current quality
        rama = geometric_validation.check_ramachandran(phi, psi)
        
        if rama['favored'] >= target_favored:
            break
        
        # Find outliers
        valid_mask = ~(np.isnan(phi) | np.isnan(psi) | ((phi == 0) & (psi == 0)))
        phi_valid = phi[valid_mask]
        psi_valid = psi[valid_mask]
        
        # Move outliers towards nearest favored region
        for i in range(len(phi_valid)):
            phi_val = phi_valid[i]
            psi_val = psi_valid[i]
            
            # Check if in favored region
            in_favored = False
            for region_name, region in geometric_validation.RAMACHANDRAN_REGIONS['favored'].items():
                if (
                    region['phi'][0] < phi_val < region['phi'][1] and
                    region['psi'][0] < psi_val < region['psi'][1]
                ):
                    in_favored = True
                    break
            
            if not in_favored:
                # Move towards center of alpha-helix region (most common)
                target_phi = -1.0  # Center of alpha-helix phi
                target_psi = -0.5  # Center of alpha-helix psi
                
                # Small step towards target
                step_size = 0.1
                phi_valid[i] += step_size * (target_phi - phi_val)
                psi_valid[i] += step_size * (target_psi - psi_val)
        
        # Update refined angles
        phi[valid_mask] = phi_valid
        psi[valid_mask] = psi_valid
        refined[:, 0] = phi
        refined[:, 1] = psi
    
    if is_torch:
        return torch.from_numpy(refined).to(device)
    return refined


def fix_omega_angles(
    angles: Union[np.ndarray, torch.Tensor],
    target_trans: float = 0.95
) -> Union[np.ndarray, torch.Tensor]:
    """
    Fix omega angles to be trans (π) for most residues.
    
    Args:
        angles: Angle tensor [n_residues, 6]
        target_trans: Target fraction of trans peptide bonds
    
    Returns:
        Angles with fixed omega values
    """
    if isinstance(angles, torch.Tensor):
        is_torch = True
        device = angles.device
        angles = angles.cpu().numpy()
    else:
        is_torch = False
    
    if angles.ndim == 1:
        angles = angles.reshape(1, -1)
    
    refined = angles.copy()
    omega = refined[:, 2]
    
    # Set omega to π (trans) for residues that are far from π
    trans_mask = np.abs(omega - np.pi) > 0.5  # More than 0.5 rad from π
    refined[trans_mask, 2] = np.pi
    
    if is_torch:
        return torch.from_numpy(refined).to(device)
    return refined


def refine_structure(
    angles: Union[np.ndarray, torch.Tensor],
    pdb_path: Optional[Union[str, Path]] = None,
    improve_ramachandran: bool = True,
    fix_omega: bool = True,
    target_rama_favored: float = 0.4
) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
    """
    Comprehensive structure refinement.
    
    Args:
        angles: Angle tensor [n_residues, 6]
        pdb_path: Optional PDB path for clash detection
        improve_ramachandran: Whether to improve Ramachandran quality
        fix_omega: Whether to fix omega angles
        target_rama_favored: Target Ramachandran favored fraction
    
    Returns:
        Tuple of (refined_angles, improvement_metrics)
    """
    if isinstance(angles, torch.Tensor):
        is_torch = True
        device = angles.device
        angles_np = angles.cpu().numpy()
    else:
        is_torch = False
        angles_np = angles.copy()
    
    # Initial quality
    initial_quality = geometric_validation.validate_structure_quality(angles_np, pdb_path)
    
    refined = angles_np.copy()
    
    # Fix omega angles
    if fix_omega:
        refined = fix_omega_angles(refined)
    
    # Improve Ramachandran quality
    if improve_ramachandran:
        refined = refine_angles_by_ramachandran(refined, target_favored=target_rama_favored)
    
    # Final quality
    final_quality = geometric_validation.validate_structure_quality(refined, pdb_path)
    
    # Improvement metrics
    improvement = {
        'initial_quality': initial_quality['quality_score'],
        'final_quality': final_quality['quality_score'],
        'quality_improvement': final_quality['quality_score'] - initial_quality['quality_score'],
        'initial_rama_favored': initial_quality['rama_favored'],
        'final_rama_favored': final_quality['rama_favored'],
        'rama_improvement': final_quality['rama_favored'] - initial_quality['rama_favored'],
        'initial_rama_outliers': initial_quality['rama_outliers'],
        'final_rama_outliers': final_quality['rama_outliers'],
        'outliers_reduction': initial_quality['rama_outliers'] - final_quality['rama_outliers'],
    }
    
    if is_torch:
        refined = torch.from_numpy(refined).to(device)
    
    return refined, improvement


def batch_refine_structures(
    angles_list: List[Union[np.ndarray, torch.Tensor]],
    pdb_paths: Optional[List[Union[str, Path]]] = None,
    min_quality_improvement: float = 0.05,
    **refinement_kwargs
) -> Tuple[List, List[Dict]]:
    """
    Refine multiple structures.
    
    Args:
        angles_list: List of angle tensors
        pdb_paths: Optional list of PDB paths
        min_quality_improvement: Minimum quality improvement to keep
        **refinement_kwargs: Arguments passed to refine_structure
    
    Returns:
        Tuple of (refined_angles_list, improvement_metrics_list)
    """
    refined_list = []
    improvements = []
    
    for i, angles in enumerate(angles_list):
        pdb_path = pdb_paths[i] if pdb_paths else None
        refined, improvement = refine_structure(angles, pdb_path=pdb_path, **refinement_kwargs)
        
        # Only keep if improvement is significant
        if improvement['quality_improvement'] >= min_quality_improvement:
            refined_list.append(refined)
            improvements.append(improvement)
        else:
            # Keep original if no significant improvement
            refined_list.append(angles)
            improvements.append({'quality_improvement': 0.0})
    
    return refined_list, improvements

