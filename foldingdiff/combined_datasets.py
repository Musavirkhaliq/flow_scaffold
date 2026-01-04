"""
Combined dataset support for training on multiple protein structure sources.

Allows combining CATH, AlphaFold, PDB, and other datasets for larger training sets.
"""
import logging
from typing import List, Optional, Union
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, Subset

from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset
from foldingdiff.enhanced_datasets import EnhancedCathDataset

logger = logging.getLogger(__name__)


class CombinedProteinDataset(Dataset):
    """
    Combined dataset from multiple sources (CATH + AlphaFold + PDB).
    
    Allows training on larger datasets by combining multiple protein structure
    databases. Supports weighted sampling from different sources.
    
    Args:
        datasets: List of dataset instances to combine
        weights: Optional weights for each dataset (for weighted sampling)
        shuffle_indices: Whether to shuffle indices across datasets
    
    Example:
        >>> cath_dset = EnhancedCathDataset(pdbs="cath", split="train")
        >>> af_dset = EnhancedCathDataset(pdbs="alphafold", split=None)
        >>> combined = CombinedProteinDataset([cath_dset, af_dset])
        >>> len(combined)  # Total structures from both datasets
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        shuffle_indices: bool = True
    ):
        if not datasets:
            raise ValueError("At least one dataset required")
        
        self.datasets = datasets
        self.weights = weights if weights else [1.0] * len(datasets)
        
        if len(self.weights) != len(datasets):
            raise ValueError(f"Weights length ({len(self.weights)}) must match datasets length ({len(datasets)})")
        
        # Compute cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for dset in datasets:
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dset)
            )
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Create shuffled index mapping if requested
        self.shuffle_indices = shuffle_indices
        if shuffle_indices:
            import numpy as np
            self.index_map = np.random.permutation(self.total_length)
        else:
            self.index_map = None
        
        logger.info(f"Combined dataset created: {self.total_length:,} total structures")
        for i, (dset, weight) in enumerate(zip(datasets, self.weights)):
            logger.info(f"  Dataset {i}: {len(dset):,} structures (weight: {weight:.2f})")
        
        # Delegate attributes from first dataset
        # (all datasets should have the same feature structure)
        if datasets:
            first_dset = datasets[0]
            if hasattr(first_dset, 'feature_names'):
                self.feature_names = first_dset.feature_names
            if hasattr(first_dset, 'feature_is_angular'):
                self.feature_is_angular = first_dset.feature_is_angular
            if hasattr(first_dset, 'pad'):
                self.pad = first_dset.pad
            if hasattr(first_dset, 'filenames'):
                # Combine filenames from all datasets
                self.filenames = []
                for dset in datasets:
                    if hasattr(dset, 'filenames'):
                        self.filenames.extend(dset.filenames)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index, ignore_zero_center: bool = False):
        """
        Get item from combined dataset.
        
        Args:
            index: Global index across all datasets
            ignore_zero_center: Whether to ignore zero-centering (passed to underlying dataset)
        
        Returns:
            Dictionary with angles, coords, etc. (same format as individual datasets)
        """
        # Map through shuffled indices if enabled
        if self.index_map is not None:
            index = int(self.index_map[index])
        
        # Find which dataset contains this index
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if index < cum_len:
                dataset_idx = i - 1
                local_idx = index - self.cumulative_lengths[i - 1]
                # Pass ignore_zero_center to underlying dataset
                dset = self.datasets[dataset_idx]
                try:
                    # Try calling with ignore_zero_center parameter
                    return dset.__getitem__(local_idx, ignore_zero_center=ignore_zero_center)
                except TypeError:
                    # Fall back to calling without the parameter if not supported
                    return dset.__getitem__(local_idx)
        
        raise IndexError(f"Index {index} out of range [0, {self.total_length})")
    
    def get_dataset_info(self) -> dict:
        """Get information about the combined dataset"""
        info = {
            'total_structures': self.total_length,
            'num_datasets': len(self.datasets),
            'datasets': []
        }
        
        for i, (dset, weight) in enumerate(zip(self.datasets, self.weights)):
            dataset_info = {
                'index': i,
                'size': len(dset),
                'weight': weight,
                'source': getattr(dset, 'pdbs_src', 'unknown')
            }
            info['datasets'].append(dataset_info)
        
        return info


def create_combined_dataset(
    cath_dir: Optional[str] = None,
    alphafold_dir: Optional[str] = None,
    pdb_dir: Optional[str] = None,
    custom_dirs: Optional[List[str]] = None,
    split: Optional[str] = "train",
    pad: int = 512,
    min_length: int = 40,
    use_enhanced: bool = True,
    force_use_cache: bool = True,  # NEW: Use cached data even if codebase hash doesn't match
    **kwargs
) -> Union[Dataset, CombinedProteinDataset]:
    """
    Create combined dataset from multiple protein structure sources.
    
    Args:
        cath_dir: Path to CATH directory (None = use default)
        alphafold_dir: Path to AlphaFold directory (None = use default or skip)
        pdb_dir: Path to PDB directory (None = skip)
        custom_dirs: List of custom directory paths to include
        split: Dataset split ('train', 'validation', 'test', or None)
        pad: Padding length
        min_length: Minimum sequence length
        use_enhanced: Whether to use EnhancedCathDataset (with coords, SS, etc.)
        **kwargs: Additional arguments passed to dataset constructors
    
    Returns:
        Combined dataset or single dataset if only one source available
    
    Example:
        >>> dataset = create_combined_dataset(
        ...     alphafold_dir="data/alphafold",
        ...     split="train",
        ...     pad=512
        ... )
        >>> len(dataset)  # Total structures from all sources
    """
    datasets = []
    dataset_weights = []
    
    # 1. CATH dataset (always try, as it's the default)
    try:
        if use_enhanced:
            cath_dset = EnhancedCathDataset(
                pdbs="cath" if cath_dir is None else cath_dir,
                split=split,
                pad=pad,
                min_length=min_length,
                force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                **kwargs
            )
        else:
            cath_dset = CathCanonicalAnglesOnlyDataset(
                pdbs="cath" if cath_dir is None else cath_dir,
                split=split,
                pad=pad,
                min_length=min_length,
                force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                **kwargs
            )
        
        if len(cath_dset) > 0:
            datasets.append(cath_dset)
            dataset_weights.append(1.0)  # Base weight
            logger.info(f"✓ CATH dataset: {len(cath_dset):,} structures")
    except Exception as e:
        logger.warning(f"CATH dataset failed: {e}")
    
    # 2. AlphaFold dataset
    if alphafold_dir is not None or Path("data/alphafold").exists():
        af_path = alphafold_dir if alphafold_dir else "data/alphafold"
        af_path = Path(af_path)
        
        if af_path.exists() and af_path.is_dir():
            # Try to load AlphaFold dataset (dataset loader will search recursively for PDB files)
            try:
                if use_enhanced:
                    af_dset = EnhancedCathDataset(
                        pdbs="alphafold" if alphafold_dir is None else str(af_path),
                        split=None,  # AlphaFold doesn't have splits - we'll split manually
                        pad=pad,
                        min_length=min_length,
                        force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                        **kwargs
                    )
                else:
                    af_dset = CathCanonicalAnglesOnlyDataset(
                        pdbs="alphafold" if alphafold_dir is None else str(af_path),
                        split=None,
                        pad=pad,
                        min_length=min_length,
                        force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                        **kwargs
                    )
                
                if len(af_dset) > 0:
                    # Manually split AlphaFold data: 80% train, 10% validation, 10% test
                    if split is not None:
                        # Use fixed seed for reproducible splits
                        rng = np.random.RandomState(seed=42)
                        indices = np.arange(len(af_dset))
                        rng.shuffle(indices)
                        
                        train_end = int(len(af_dset) * 0.8)
                        val_end = int(len(af_dset) * 0.9)
                        
                        if split == "train":
                            indices = indices[:train_end]
                            logger.info(f"✓ AlphaFold dataset (train): {len(indices):,} structures (80%)")
                        elif split == "validation":
                            indices = indices[train_end:val_end]
                            logger.info(f"✓ AlphaFold dataset (validation): {len(indices):,} structures (10%)")
                        elif split == "test":
                            indices = indices[val_end:]
                            logger.info(f"✓ AlphaFold dataset (test): {len(indices):,} structures (10%)")
                        else:
                            indices = indices  # Use all for None
                        
                        # Create subset
                        af_dset = Subset(af_dset, indices.tolist())
                    else:
                        logger.info(f"✓ AlphaFold dataset: {len(af_dset):,} structures (no split)")
                    
                    datasets.append(af_dset)
                    dataset_weights.append(1.0)  # Same weight as CATH
            except Exception as e:
                logger.warning(f"AlphaFold dataset failed: {e}")
    
    # 3. PDB dataset (if directory provided)
    if pdb_dir is not None:
        pdb_path = Path(pdb_dir)
        if pdb_path.exists():
            pdb_files = list(pdb_path.glob("*.pdb*"))
            if pdb_files:
                try:
                    if use_enhanced:
                        pdb_dset = EnhancedCathDataset(
                            pdbs=str(pdb_path),
                            split=None,  # PDB doesn't have splits - we'll split manually
                            pad=pad,
                            min_length=min_length,
                            force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                            **kwargs
                        )
                    else:
                        pdb_dset = CathCanonicalAnglesOnlyDataset(
                            pdbs=str(pdb_path),
                            split=None,
                            pad=pad,
                            min_length=min_length,
                            force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                            **kwargs
                        )
                    
                    if len(pdb_dset) > 0:
                        # Manually split PDB data: 80% train, 10% validation, 10% test
                        if split is not None:
                            rng = np.random.RandomState(seed=42)
                            indices = np.arange(len(pdb_dset))
                            rng.shuffle(indices)
                            
                            train_end = int(len(pdb_dset) * 0.8)
                            val_end = int(len(pdb_dset) * 0.9)
                            
                            if split == "train":
                                indices = indices[:train_end]
                                logger.info(f"✓ PDB dataset (train): {len(indices):,} structures (80%)")
                            elif split == "validation":
                                indices = indices[train_end:val_end]
                                logger.info(f"✓ PDB dataset (validation): {len(indices):,} structures (10%)")
                            elif split == "test":
                                indices = indices[val_end:]
                                logger.info(f"✓ PDB dataset (test): {len(indices):,} structures (10%)")
                            else:
                                indices = indices
                            
                            pdb_dset = Subset(pdb_dset, indices.tolist())
                        else:
                            logger.info(f"✓ PDB dataset: {len(pdb_dset):,} structures (no split)")
                        
                        datasets.append(pdb_dset)
                        dataset_weights.append(1.0)
                except Exception as e:
                    logger.warning(f"PDB dataset failed: {e}")
    
    # 4. Custom directories
    if custom_dirs:
        for custom_dir in custom_dirs:
            custom_path = Path(custom_dir)
            if custom_path.exists():
                pdb_files = list(custom_path.glob("*.pdb*"))
                if pdb_files:
                    try:
                        if use_enhanced:
                            custom_dset = EnhancedCathDataset(
                                pdbs=str(custom_path),
                                split=None,  # Custom dirs don't have splits - we'll split manually
                                pad=pad,
                                min_length=min_length,
                                force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                                **kwargs
                            )
                        else:
                            custom_dset = CathCanonicalAnglesOnlyDataset(
                                pdbs=str(custom_path),
                                split=None,
                                pad=pad,
                                min_length=min_length,
                                force_use_cache=force_use_cache,  # NEW: Pass through force_use_cache
                                **kwargs
                            )
                        
                        if len(custom_dset) > 0:
                            # Manually split custom data: 80% train, 10% validation, 10% test
                            if split is not None:
                                rng = np.random.RandomState(seed=42)
                                indices = np.arange(len(custom_dset))
                                rng.shuffle(indices)
                                
                                train_end = int(len(custom_dset) * 0.8)
                                val_end = int(len(custom_dset) * 0.9)
                                
                                if split == "train":
                                    indices = indices[:train_end]
                                    logger.info(f"✓ Custom dataset ({custom_dir}) (train): {len(indices):,} structures (80%)")
                                elif split == "validation":
                                    indices = indices[train_end:val_end]
                                    logger.info(f"✓ Custom dataset ({custom_dir}) (validation): {len(indices):,} structures (10%)")
                                elif split == "test":
                                    indices = indices[val_end:]
                                    logger.info(f"✓ Custom dataset ({custom_dir}) (test): {len(indices):,} structures (10%)")
                                else:
                                    indices = indices
                                
                                custom_dset = Subset(custom_dset, indices.tolist())
                            else:
                                logger.info(f"✓ Custom dataset ({custom_dir}): {len(custom_dset):,} structures (no split)")
                            
                            datasets.append(custom_dset)
                            dataset_weights.append(1.0)
                    except Exception as e:
                        logger.warning(f"Custom dataset ({custom_dir}) failed: {e}")
    
    if not datasets:
        raise ValueError(
            "No datasets available! Please ensure at least one of:\n"
            "- CATH dataset is available (data/cath/dompdb)\n"
            "- AlphaFold dataset is available (data/alphafold)\n"
            "- PDB dataset is available (data/pdb)\n"
            "- Custom directories are provided"
        )
    
    # Return single dataset if only one, otherwise combined
    if len(datasets) == 1:
        logger.info(f"Using single dataset: {len(datasets[0]):,} structures")
        return datasets[0]
    else:
        combined = CombinedProteinDataset(datasets, weights=dataset_weights)
        logger.info(f"✓ Combined dataset: {len(combined):,} total structures from {len(datasets)} sources")
        return combined


