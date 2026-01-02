"""
Utilities for saving training metadata (means, etc.) during training.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def save_training_means(
    means: np.ndarray,
    output_dir: Path,
    filename: str = "training_mean_offset.npy"
) -> Path:
    """
    Save training means to file.
    
    Should be called during training after dataset initialization.
    
    Args:
        means: Mean values array [n_features]
        output_dir: Directory to save to
        filename: Filename (default: training_mean_offset.npy)
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / filename
    np.save(output_file, means)
    
    logger.info(f"✓ Saved training means to {output_file}")
    logger.info(f"  Means: {means}")
    
    return output_file


def save_training_metadata(
    dataset,
    output_dir: Path,
    save_means: bool = True
) -> dict:
    """
    Save training metadata from dataset.
    
    Args:
        dataset: Dataset instance (should have .means attribute)
        output_dir: Directory to save to
        save_means: Whether to save means
    
    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if save_means and hasattr(dataset, 'means') and dataset.means is not None:
        means_file = save_training_means(dataset.means, output_dir)
        saved_files['means'] = means_file
    
    return saved_files

