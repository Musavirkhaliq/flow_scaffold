"""
Geometric validation utilities for protein structures.

Provides functions to:
- Validate Ramachandran angles
- Detect steric clashes
- Check angle ranges
- Validate geometric constraints
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

logger = logging.getLogger(__name__)


# Ramachandran plot regions (in radians)
RAMACHANDRAN_REGIONS = {
    'favored': {
        'alpha': {'phi': (-2.0, -0.5), 'psi': (-1.5, 0.5)},
        'beta': {'phi': (-2.5, -0.5), 'psi': (1.0, 2.5)},
        'ppii': {'phi': (-1.5, 0.0), 'psi': (0.5, 2.0)},
    },
    'allowed': {
        'alpha': {'phi': (-2.5, -0.3), 'psi': (-2.0, 1.0)},
        'beta': {'phi': (-3.0, -0.3), 'psi': (0.5, 3.0)},
        'ppii': {'phi': (-2.0, 0.5), 'psi': (0.0, 2.5)},
    }
}


def check_ramachandran(
    phi: Union[np.ndarray, torch.Tensor],
    psi: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Check Ramachandran plot quality.
    
    Args:
        phi: Phi angles in radians [n_residues]
        psi: Psi angles in radians [n_residues]
    
    Returns:
        Dictionary with:
        - favored: Fraction in favored regions
        - allowed: Fraction in allowed regions
        - outliers: Fraction of outliers
        - alpha_fraction: Fraction in alpha-helix region
        - beta_fraction: Fraction in beta-sheet region
        - ppii_fraction: Fraction in PPII region
    """
    # Convert to numpy if needed
    if isinstance(phi, torch.Tensor):
        phi = phi.cpu().numpy()
    if isinstance(psi, torch.Tensor):
        psi = psi.cpu().numpy()
    
    # Remove NaN and padding (zeros)
    valid_mask = ~(np.isnan(phi) | np.isnan(psi) | ((phi == 0) & (psi == 0)))
    phi_valid = phi[valid_mask]
    psi_valid = psi[valid_mask]
    
    if len(phi_valid) == 0:
        return {
            'favored': 0.0,
            'allowed': 0.0,
            'outliers': 1.0,
            'alpha_fraction': 0.0,
            'beta_fraction': 0.0,
            'ppii_fraction': 0.0,
        }
    
    n_residues = len(phi_valid)
    
    # Check favored regions
    alpha_favored = (
        (phi_valid > RAMACHANDRAN_REGIONS['favored']['alpha']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['favored']['alpha']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['favored']['alpha']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['favored']['alpha']['psi'][1])
    )
    
    beta_favored = (
        (phi_valid > RAMACHANDRAN_REGIONS['favored']['beta']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['favored']['beta']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['favored']['beta']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['favored']['beta']['psi'][1])
    )
    
    ppii_favored = (
        (phi_valid > RAMACHANDRAN_REGIONS['favored']['ppii']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['favored']['ppii']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['favored']['ppii']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['favored']['ppii']['psi'][1])
    )
    
    favored = (alpha_favored | beta_favored | ppii_favored).sum() / n_residues
    
    # Check allowed regions
    alpha_allowed = (
        (phi_valid > RAMACHANDRAN_REGIONS['allowed']['alpha']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['allowed']['alpha']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['allowed']['alpha']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['allowed']['alpha']['psi'][1])
    )
    
    beta_allowed = (
        (phi_valid > RAMACHANDRAN_REGIONS['allowed']['beta']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['allowed']['beta']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['allowed']['beta']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['allowed']['beta']['psi'][1])
    )
    
    ppii_allowed = (
        (phi_valid > RAMACHANDRAN_REGIONS['allowed']['ppii']['phi'][0]) &
        (phi_valid < RAMACHANDRAN_REGIONS['allowed']['ppii']['phi'][1]) &
        (psi_valid > RAMACHANDRAN_REGIONS['allowed']['ppii']['psi'][0]) &
        (psi_valid < RAMACHANDRAN_REGIONS['allowed']['ppii']['psi'][1])
    )
    
    allowed = (alpha_allowed | beta_allowed | ppii_allowed).sum() / n_residues
    outliers = 1.0 - allowed
    
    # Secondary structure fractions
    alpha_fraction = alpha_favored.sum() / n_residues
    beta_fraction = beta_favored.sum() / n_residues
    ppii_fraction = ppii_favored.sum() / n_residues
    
    return {
        'favored': float(favored),
        'allowed': float(allowed),
        'outliers': float(outliers),
        'alpha_fraction': float(alpha_fraction),
        'beta_fraction': float(beta_fraction),
        'ppii_fraction': float(ppii_fraction),
    }


def validate_angles(
    angles: Union[np.ndarray, torch.Tensor],
    angle_names: List[str] = ['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA']
) -> Dict[str, Union[bool, float]]:
    """
    Validate that angles are within reasonable ranges.
    
    Args:
        angles: Angle tensor [n_residues, n_angles] or [n_angles]
        angle_names: Names of angles
    
    Returns:
        Dictionary with validation results
    """
    if isinstance(angles, torch.Tensor):
        angles = angles.cpu().numpy()
    
    if angles.ndim == 1:
        angles = angles.reshape(1, -1)
    
    results = {
        'all_valid': True,
        'phi_in_range': True,
        'psi_in_range': True,
        'omega_in_range': True,
        'tau_in_range': True,
        'out_of_range_fraction': 0.0,
    }
    
    # Expected ranges (in radians)
    ranges = {
        'phi': (-np.pi, np.pi),
        'psi': (-np.pi, np.pi),
        'omega': (-np.pi, np.pi),  # Usually ~π (trans) or 0 (cis)
        'tau': (1.5, 2.3),  # ~85-130 degrees
        'CA:C:1N': (1.8, 2.3),  # ~103-132 degrees
        'C:1N:1CA': (1.9, 2.4),  # ~109-138 degrees
    }
    
    n_residues = angles.shape[0]
    out_of_range = np.zeros(n_residues, dtype=bool)
    
    for i, name in enumerate(angle_names):
        if name not in ranges:
            continue
        
        angle_values = angles[:, i]
        min_val, max_val = ranges[name]
        
        # Check range
        in_range = (angle_values >= min_val) & (angle_values <= max_val)
        out_of_range |= ~in_range
        
        results[f'{name}_in_range'] = bool(in_range.all())
    
    results['out_of_range_fraction'] = float(out_of_range.sum() / n_residues)
    results['all_valid'] = results['out_of_range_fraction'] < 0.1
    
    return results


def detect_clashes_from_pdb(
    pdb_path: Union[str, Path],
    clash_distance: float = 2.0
) -> Dict[str, Union[int, float]]:
    """
    Detect Van der Waals clashes in a PDB structure.
    
    Args:
        pdb_path: Path to PDB file
        clash_distance: Minimum distance for clash (Angstroms)
    
    Returns:
        Dictionary with clash information
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return {'n_clashes': 0, 'clash_rate': 0.0, 'error': 'File not found'}
    
    try:
        struct = PDBFile.read(str(pdb_path)).get_structure()[0]
        
        # Get all atom coordinates
        coords = struct.coord
        n_atoms = len(coords)
        
        if n_atoms < 2:
            return {'n_clashes': 0, 'clash_rate': 0.0}
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(coords))
        
        # Find clashes (too close, but not same atom)
        np.fill_diagonal(distances, np.inf)
        clashes = (distances < clash_distance) & (distances > 0)
        n_clashes = int(clashes.sum() / 2)  # Each pair counted twice
        
        # Clash rate per atom
        clash_rate = n_clashes / (n_atoms * (n_atoms - 1) / 2) if n_atoms > 1 else 0.0
        
        return {
            'n_clashes': n_clashes,
            'clash_rate': float(clash_rate),
            'n_atoms': n_atoms,
        }
    
    except Exception as e:
        logger.error(f"Error detecting clashes in {pdb_path}: {e}")
        return {'n_clashes': 0, 'clash_rate': 0.0, 'error': str(e)}


def validate_structure_quality(
    angles: Union[np.ndarray, torch.Tensor],
    pdb_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[bool, float, int]]:
    """
    Comprehensive structure quality validation.
    
    Args:
        angles: Angle tensor [n_residues, 6]
        pdb_path: Optional PDB file path for clash detection
    
    Returns:
        Dictionary with all quality metrics
    """
    if isinstance(angles, torch.Tensor):
        angles = angles.cpu().numpy()
    
    if angles.ndim == 1:
        angles = angles.reshape(1, -1)
    
    results = {}
    
    # Extract angles
    phi = angles[:, 0]
    psi = angles[:, 1]
    omega = angles[:, 2]
    tau = angles[:, 3] if angles.shape[1] > 3 else None
    
    # Ramachandran check
    ramachandran = check_ramachandran(phi, psi)
    results.update({f'rama_{k}': v for k, v in ramachandran.items()})
    
    # Angle validation
    angle_validation = validate_angles(angles)
    results.update(angle_validation)
    
    # Omega check (should be ~π for trans)
    omega_trans = np.abs(omega - np.pi) < 0.5  # Within 0.5 rad of π
    results['omega_trans_fraction'] = float(omega_trans.sum() / len(omega))
    
    # Tau check (should be ~1.92 rad = 110°)
    if tau is not None:
        tau_valid = (tau > 1.5) & (tau < 2.3)
        results['tau_valid_fraction'] = float(tau_valid.sum() / len(tau))
    
    # Clash detection if PDB provided
    if pdb_path:
        clashes = detect_clashes_from_pdb(pdb_path)
        results.update(clashes)
    
    # Overall quality score (weighted combination)
    quality_score = (
        0.4 * ramachandran['favored'] +
        0.3 * (1.0 - ramachandran['outliers']) +
        0.2 * angle_validation['all_valid'] +
        0.1 * (1.0 - results.get('clash_rate', 0.0))
    )
    results['quality_score'] = float(quality_score)
    
    # Pass/fail
    results['passes_validation'] = (
        ramachandran['favored'] > 0.3 and
        ramachandran['outliers'] < 0.5 and
        results.get('clash_rate', 0.0) < 0.2
    )
    
    return results


def filter_by_quality(
    angles_list: List[Union[np.ndarray, torch.Tensor]],
    min_quality: float = 0.3,
    min_rama_favored: float = 0.2,
    max_rama_outliers: float = 0.6
) -> Tuple[List, List[int]]:
    """
    Filter structures by quality metrics.
    
    Args:
        angles_list: List of angle tensors
        min_quality: Minimum quality score
        min_rama_favored: Minimum Ramachandran favored fraction
        max_rama_outliers: Maximum Ramachandran outliers fraction
    
    Returns:
        Tuple of (filtered_angles, kept_indices)
    """
    filtered = []
    kept_indices = []
    
    for i, angles in enumerate(angles_list):
        quality = validate_structure_quality(angles)
        
        if (
            quality['quality_score'] >= min_quality and
            quality['rama_favored'] >= min_rama_favored and
            quality['rama_outliers'] <= max_rama_outliers
        ):
            filtered.append(angles)
            kept_indices.append(i)
    
    return filtered, kept_indices

