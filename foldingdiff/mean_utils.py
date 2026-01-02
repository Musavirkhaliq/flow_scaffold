"""
Utilities for computing and loading training means.

Handles mean computation with proper handling of angular vs non-angular features.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pickle

from foldingdiff import custom_metrics as cm
from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset

logger = logging.getLogger(__name__)

# Cache for computed means (avoid recomputing)
_MEANS_CACHE = {}


def compute_training_means(
    pdbs: str = "cath",
    split: Optional[str] = None,
    pad: int = 512,
    min_length: int = 40,
    zero_center: bool = True,
    cache_key: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Compute training means from CATH dataset.
    
    Uses wrapped_mean for angular features (phi, psi, omega) and
    regular mean for bond angles (tau, CA:C:1N, C:1N:1CA).
    
    Args:
        pdbs: PDB source ("cath" or path)
        split: Dataset split (None = all)
        pad: Padding length
        min_length: Minimum sequence length
        zero_center: Whether to zero-center (should match training)
        cache_key: Optional cache key to avoid recomputation
    
    Returns:
        Array of means [phi, psi, omega, tau, CA:C:1N, C:1N:1CA] or None
    """
    # Check cache
    if cache_key and cache_key in _MEANS_CACHE:
        logger.info(f"✓ Using cached means for {cache_key}")
        return _MEANS_CACHE[cache_key]
    
    try:
        logger.info("Computing training means from CATH dataset...")
        logger.info(f"  Parameters: pdbs={pdbs}, split={split}, pad={pad}, min_length={min_length}")
        
        # Create dataset (this will compute means)
        dataset = CathCanonicalAnglesOnlyDataset(
            pdbs=pdbs,
            split=split,
            pad=pad,
            min_length=min_length,
            zero_center=zero_center
        )
        
        if dataset.means is None:
            logger.error("Dataset means are None (zero_center=False?)")
            return None
        
        means = dataset.means.copy()
        
        # Verify shape
        expected_features = 6  # phi, psi, omega, tau, CA:C:1N, C:1N:1CA
        if means.shape[0] != expected_features:
            logger.warning(
                f"Means shape mismatch: expected {expected_features}, got {means.shape[0]}. "
                f"Using first {expected_features} features."
            )
            means = means[:expected_features]
        
        # Cache result
        if cache_key:
            _MEANS_CACHE[cache_key] = means
        
        logger.info(f"✓ Computed means: phi={means[0]:.3f}, psi={means[1]:.3f}, "
                   f"omega={means[2]:.3f}, tau={means[3]:.3f}, "
                   f"CA:C:1N={means[4]:.3f}, C:1N:1CA={means[5]:.3f}")
        
        return means
    
    except Exception as e:
        logger.error(f"Failed to compute means from dataset: {e}")
        return None


def load_training_means(
    model_dir: Path,
    pdbs: str = "cath",
    split: Optional[str] = None,
    pad: int = 512,
    min_length: int = 40
) -> Tuple[Optional[np.ndarray], str]:
    """
    Load training means with best-effort fallback.
    
    Priority:
    1. Load from training_mean_offset.npy (if exists)
    2. Compute from dataset (cached)
    3. Fallback to hardcoded values (last resort)
    
    Args:
        model_dir: Model directory path
        pdbs: PDB source for dataset computation
        split: Dataset split
        pad: Padding length
        min_length: Minimum sequence length
    
    Returns:
        Tuple of (means_array, source) where source is:
        - "file" if loaded from file
        - "dataset" if computed from dataset
        - "hardcoded" if using fallback
    """
    model_dir = Path(model_dir)
    
    # Priority 1: Load from file
    mean_offset_file = model_dir / "training_mean_offset.npy"
    if mean_offset_file.exists():
        try:
            means = np.load(mean_offset_file)
            logger.info(f"✓ Loaded training means from {mean_offset_file}")
            logger.info(f"  Means: phi={means[0]:.3f}, psi={means[1]:.3f}, "
                       f"omega={means[2]:.3f}, tau={means[3]:.3f}")
            return means, "file"
        except Exception as e:
            logger.warning(f"Failed to load means from file: {e}")
    
    # Priority 2: Compute from dataset
    cache_key = f"{pdbs}_{split}_{pad}_{min_length}"
    means = compute_training_means(
        pdbs=pdbs,
        split=split,
        pad=pad,
        min_length=min_length,
        cache_key=cache_key
    )
    
    if means is not None:
        # Try to save for future use
        try:
            mean_offset_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(mean_offset_file, means)
            logger.info(f"✓ Saved computed means to {mean_offset_file} for future use")
        except Exception as e:
            logger.debug(f"Could not save means to file: {e}")
        
        return means, "dataset"
    
    # Priority 3: Fallback to hardcoded (last resort)
    logger.warning("⚠️  Using hardcoded means (may be incorrect!)")
    logger.warning("  This may cause poor geometric quality!")
    logger.warning("  Consider saving means during training or ensuring dataset is available")
    
    # Hardcoded fallback values
    # These are approximate and may not match actual training data
    hardcoded_means = np.array([
        0.0,      # phi: typically centered around 0
        0.0,      # psi: typically centered around 0
        np.pi,    # omega: ~π for trans peptide bonds
        1.92,     # tau: ~110° (N-CA-C bond angle)
        2.01,     # CA:C:1N: ~115° (C-N bond angle)
        2.11,     # C:1N:1CA: ~121° (N-CA bond angle)
    ])
    
    return hardcoded_means, "hardcoded"


def apply_means_with_wrapping(
    angles: np.ndarray,
    means: np.ndarray,
    is_angular: Optional[list] = None,
    feature_idx: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply means to angles with proper wrapping for angular features.
    
    Args:
        angles: Angle array [n_residues, n_features] (typically 6 features)
        means: Mean values [n_features] (may be 9 features from full dataset)
        is_angular: List indicating which features are angular (default: [True, True, True, False, False, False])
        feature_idx: Optional indices to subset means if means has more features than angles
    
    Returns:
        Angles with means added, with angular features wrapped to [-π, π]
    """
    if is_angular is None:
        is_angular = [True, True, True, False, False, False]  # phi, psi, omega, tau, CA:C:1N, C:1N:1CA
    
    # Handle case where means has more features than angles (e.g., 9 vs 6)
    # This happens when means come from full canonical dataset but we only use 6 features
    n_angle_features = angles.shape[1] if angles.ndim > 1 else len(angles)
    n_mean_features = means.shape[0] if means.ndim > 0 else len(means)
    
    if n_mean_features > n_angle_features:
        # Means come from full dataset (9 features), need to subset to 6 features
        # Feature order in full dataset: [0C:1N, N:CA, CA:C, phi, psi, omega, tau, CA:C:1N, C:1N:1CA]
        # We want: [phi, psi, omega, tau, CA:C:1N, C:1N:1CA] = indices [3, 4, 5, 6, 7, 8]
        if feature_idx is None:
            # Default: assume we're using the 6 angle features (indices 3-8)
            feature_idx = np.array([3, 4, 5, 6, 7, 8])
        means = means[feature_idx]
        logger.debug(f"Subset means from {n_mean_features} to {len(means)} features using indices {feature_idx}")
    elif n_mean_features < n_angle_features:
        raise ValueError(f"Means has fewer features ({n_mean_features}) than angles ({n_angle_features})")
    
    # Ensure means matches angles shape
    if angles.ndim == 1:
        angles = angles.reshape(1, -1)
    
    # Add means
    angles_corrected = angles + means
    
    # Wrap angular features
    for j, angular in enumerate(is_angular):
        if angular:
            angles_corrected[:, j] = np.arctan2(
                np.sin(angles_corrected[:, j]),
                np.cos(angles_corrected[:, j])
            )
    
    return angles_corrected

