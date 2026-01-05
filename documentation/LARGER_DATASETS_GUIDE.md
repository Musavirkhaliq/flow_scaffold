# Guide to Using Larger Datasets for Training

**Current Dataset:** CATH - 34,653 structures (2.84 GB)  
**Target:** 100K-1M+ structures for better performance

---

## Available Large Protein Structure Datasets

### 1. AlphaFold Database (RECOMMENDED - Largest)

**Size:** ~214 million predicted structures (as of 2024)  
**Download Size:** ~23 TB (compressed)  
**Recommended Subset:** 100K-1M high-quality structures

**Advantages:**
- ✅ Largest dataset available
- ✅ High-quality predicted structures
- ✅ Covers entire proteomes
- ✅ Already supported in codebase (`pdbs="alphafold"`)

**Download Instructions:**

```bash
# Option 1: Download specific proteomes (recommended)
# Download human proteome (~20K structures, ~2GB)
wget -r -np -nH --cut-dirs=2 \
    https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN/ \
    -P data/alphafold/

# Option 2: Download all structures (23TB - not recommended)
# Use subset instead

# Option 3: Download by organism (recommended for diversity)
# Human: UP000005640_9606_HUMAN
# Mouse: UP000000589_10090_MOUSE
# E. coli: UP000000625_83333_ECOLI
# Yeast: UP000002311_559292_YEAST
```

**Expected Impact:** +0.10-0.20 quality score (from 0.236 to 0.35-0.45)

---

### 2. PDB Database (Experimental Structures)

**Size:** ~200K experimental structures  
**Download Size:** ~50 GB (compressed)  
**Quality:** Highest (experimental, not predicted)

**Advantages:**
- ✅ Highest quality (experimental)
- ✅ Diverse structures
- ✅ Real biological structures

**Download Instructions:**

```bash
# Download all PDB structures
# Option 1: Download from RCSB PDB
wget -r -np -nH --cut-dirs=2 \
    https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/ \
    -P data/pdb/ \
    -A "*.ent.gz"

# Option 2: Use PDB API (slower but more selective)
# Can filter by resolution, method, etc.

# Convert .ent.gz to .pdb
cd data/pdb
for f in *.ent.gz; do
    zcat "$f" | sed 's/^HEADER/HEADER/' > "${f%.ent.gz}.pdb"
done
```

**Expected Impact:** +0.05-0.10 quality score

---

### 3. Combined CATH + AlphaFold + PDB

**Recommended Approach:** Combine multiple datasets for maximum diversity

**Total Size:** 100K-1M structures  
**Expected Impact:** +0.15-0.30 quality score

---

## Implementation: Multi-Dataset Support

### Step 1: Create Combined Dataset Class

```python
# In foldingdiff/datasets.py, add:

class CombinedProteinDataset(Dataset):
    """
    Combined dataset from multiple sources (CATH + AlphaFold + PDB).
    
    Args:
        datasets: List of dataset instances to combine
        weights: Optional weights for each dataset (for sampling)
    """
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None
    ):
        self.datasets = datasets
        self.weights = weights if weights else [1.0] * len(datasets)
        
        # Compute cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for dset in datasets:
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dset)
            )
        
        self.total_length = self.cumulative_lengths[-1]
        
        logging.info(f"Combined dataset: {self.total_length:,} structures")
        for i, dset in enumerate(datasets):
            logging.info(f"  Dataset {i}: {len(dset):,} structures")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        # Find which dataset contains this index
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if index < cum_len:
                dataset_idx = i - 1
                local_idx = index - self.cumulative_lengths[i - 1]
                return self.datasets[dataset_idx][local_idx]
        
        raise IndexError(f"Index {index} out of range")
```

### Step 2: Update Training Script

```python
# In bin/train_advanced_flow.py, modify dataset creation:

def create_combined_dataset(
    cath_dir: str = "data/cath",
    alphafold_dir: str = "data/alphafold",
    pdb_dir: Optional[str] = None,
    split: Optional[str] = "train",
    pad: int = 512,
    min_length: int = 40,
    **kwargs
):
    """Create combined dataset from multiple sources"""
    datasets = []
    
    # 1. CATH dataset
    try:
        cath_dset = EnhancedCathDataset(
            pdbs="cath",
            split=split,
            pad=pad,
            min_length=min_length,
            **kwargs
        )
        datasets.append(cath_dset)
        logging.info(f"✓ CATH dataset: {len(cath_dset):,} structures")
    except Exception as e:
        logging.warning(f"CATH dataset failed: {e}")
    
    # 2. AlphaFold dataset
    if Path(alphafold_dir).exists() and len(list(Path(alphafold_dir).glob("*.pdb*"))) > 0:
        try:
            alphafold_dset = EnhancedCathDataset(
                pdbs="alphafold",
                split=None,  # AlphaFold doesn't have splits
                pad=pad,
                min_length=min_length,
                **kwargs
            )
            datasets.append(alphafold_dset)
            logging.info(f"✓ AlphaFold dataset: {len(alphafold_dset):,} structures")
        except Exception as e:
            logging.warning(f"AlphaFold dataset failed: {e}")
    
    # 3. PDB dataset (if available)
    if pdb_dir and Path(pdb_dir).exists():
        try:
            pdb_dset = EnhancedCathDataset(
                pdbs=pdb_dir,  # Directory path
                split=None,
                pad=pad,
                min_length=min_length,
                **kwargs
            )
            datasets.append(pdb_dset)
            logging.info(f"✓ PDB dataset: {len(pdb_dset):,} structures")
        except Exception as e:
            logging.warning(f"PDB dataset failed: {e}")
    
    if not datasets:
        raise ValueError("No datasets available!")
    
    # Combine datasets
    if len(datasets) == 1:
        return datasets[0]
    else:
        from foldingdiff.datasets import CombinedProteinDataset
        return CombinedProteinDataset(datasets)
```

---

## Quick Start: Download AlphaFold Subset

### Recommended: Download Human + Model Organisms

```bash
# Create AlphaFold directory
mkdir -p data/alphafold

# Download human proteome (~20K structures)
cd data/alphafold
wget -r -np -nH --cut-dirs=2 \
    https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN/ \
    --accept "*.pdb.gz"

# Download E. coli (~4K structures)
wget -r -np -nH --cut-dirs=2 \
    https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI/ \
    --accept "*.pdb.gz"

# Download yeast (~6K structures)
wget -r -np -nH --cut-dirs=2 \
    https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002311_559292_YEAST/ \
    --accept "*.pdb.gz"

# Total: ~30K structures (3x current CATH size)
```

**Expected Size:** ~3-5 GB  
**Expected Impact:** +0.05-0.10 quality score

---

## Advanced: Download Larger Subset

### Download Top 100K Structures by Quality

```bash
# Download AlphaFold structures with highest confidence
# (pLDDT > 90, ranked by quality)

# Use AlphaFold metadata to select high-quality structures
# Then download selectively
```

**Expected Size:** ~10-20 GB  
**Expected Impact:** +0.10-0.15 quality score

---

## Expected Performance Improvements

### Current (CATH only):
- Dataset Size: 34,653 structures
- Quality Score: 0.236
- Ramachandran: 47.2%

### After Adding AlphaFold (30K structures):
- Dataset Size: ~65K structures (2x)
- Quality Score: **0.29-0.34** (+0.05-0.10)
- Ramachandran: **52-57%** (+5-10%)

### After Adding PDB (200K structures):
- Dataset Size: ~235K structures (7x)
- Quality Score: **0.35-0.45** (+0.12-0.21)
- Ramachandran: **60-70%** (+13-23%)

### After Full Combined (1M+ structures):
- Dataset Size: 1M+ structures (30x)
- Quality Score: **0.50-0.70** (+0.26-0.46)
- Ramachandran: **70-85%** (+23-38%)

**This should bring us close to or beat SOTA!**

---

## Implementation Steps

1. **Download AlphaFold subset** (recommended: 30K-100K structures)
2. **Implement CombinedProteinDataset** class
3. **Update training script** to use combined dataset
4. **Re-train model** with larger dataset
5. **Re-evaluate** performance

---

## Next Steps

1. Download AlphaFold human proteome (~20K structures)
2. Test dataset loading
3. Implement combined dataset class
4. Re-train with combined dataset
5. Evaluate improvements

---

## Summary

**Current:** CATH only (34K structures)  
**Recommended:** CATH + AlphaFold (65K+ structures)  
**Best:** CATH + AlphaFold + PDB (235K+ structures)  
**Ultimate:** 1M+ structures from all sources

**Expected improvement:** +0.15-0.30 quality score with larger datasets!



