# Reference Structures Guide

## What Are Reference Structures?

Reference structures are **ground truth** protein structures used to compare against your generated structures. They enable computation of:

1. **Structural Similarity** - How similar are your structures to the references? (RMSD, TM-score, GDT)
2. **Motif Recovery** - How well are motifs preserved? (motif RMSD, motif TM-score)
3. **Sequence-Structure Compatibility** - How well do sequences match structures?

---

## Where to Find Reference Structures

### Option 1: CATH Database (Recommended for Novelty/Diversity)

Your project already has the **CATH database** installed:

**Location:** `/disk-10tb/flow_scaffold/data/cath/dompdb/`

**Contents:** 34,653 protein domain structures (no file extensions)

**Usage:** Use for novelty evaluation (checking if generated structures are novel vs. existing database)

```bash
# Use CATH database for novelty evaluation
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \
    --database_dir data/cath/dompdb
```

### Option 2: Original Training/Test Structures

For **motif scaffolding**, the best references are the **original structures** that contain the motifs you're trying to scaffold.

**Where to find them:**

1. **CATH Test Set** - If you used a train/test split:
   ```python
   # In your training code, you likely have:
   test_dataset = CathCanonicalAnglesDataset(
       pdbs="cath",
       split="test",  # or "validation"
       ...
   )
   ```

2. **Specific CATH Structures** - If you know which CATH structures contain your motifs:
   ```bash
   # Example: Find structures in CATH
   ls data/cath/dompdb/ | grep -i "your_pattern"
   ```

3. **PDB Database** - Download from RCSB PDB:
   ```bash
   # Download specific structures
   wget https://files.rcsb.org/download/1ABC.pdb -O data/references/1ABC.pdb
   ```

### Option 3: Create Reference Directory from CATH

You can extract specific structures from CATH to use as references:

```bash
# Create reference directory
mkdir -p data/references

# Copy specific CATH structures (example)
cp data/cath/dompdb/12asA00 data/references/
cp data/cath/dompdb/152lA00 data/references/
cp data/cath/dompdb/153lA00 data/references/

# Or copy a subset
ls data/cath/dompdb/ | head -100 | while read f; do
    cp "data/cath/dompdb/$f" "data/references/${f}.pdb"
done
```

---

## How to Organize Reference Structures

### Directory Structure

Create a directory with PDB files:

```
data/
└── references/
    ├── reference_0000.pdb
    ├── reference_0001.pdb
    ├── reference_0002.pdb
    └── ...
```

### File Naming

The evaluation script matches generated structures to references using **two strategies**:

#### Strategy 1: Name-based Matching (Preferred)

Match by filename (without extension):

- Generated: `sample_0000.pdb` → Matches → Reference: `sample_0000.pdb` or `reference_0000.pdb`
- Generated: `sample_0001.pdb` → Matches → Reference: `sample_0001.pdb` or `reference_0001.pdb`

**Best practice:** Use the same naming pattern:
```bash
# Rename references to match generated structures
cd data/references
for i in {0..24}; do
    mv "cath_structure_${i}" "sample_$(printf "%04d" $i).pdb"
done
```

#### Strategy 2: Index-based Matching (Fallback)

If name matching fails, the script matches by index order:
- `sample_0000.pdb` → `reference_0000.pdb` (1st file)
- `sample_0001.pdb` → `reference_0001.pdb` (2nd file)
- etc.

**Important:** Ensure the same number of generated and reference structures!

---

## How to Use Reference Structures

### Basic Usage

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \
    --reference_dir data/references
```

### With Specific Scenario

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234 \
    --reference_dir data/references \
    --scenario two_motifs_short
```

### Using CATH Database for Novelty

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \
    --database_dir data/cath/dompdb
```

---

## What Reference Structures Do You Need?

### For Motif Scaffolding Evaluation

If you're doing **motif scaffolding** (like `two_motifs_short`), you need:

1. **Original structures** that contain the motifs
2. **Same motif regions** as in your generated structures
3. **Same number** of structures (or at least matching pairs)

**Example workflow:**

```python
# 1. Identify which CATH structures contain your motifs
# (This depends on your training data)

# 2. Extract those structures from CATH
import shutil
from pathlib import Path

cath_dir = Path("data/cath/dompdb")
ref_dir = Path("data/references")
ref_dir.mkdir(exist_ok=True)

# Example: Copy structures that match your motif patterns
motif_containing_structures = [
    "12asA00",  # Replace with actual structure IDs
    "152lA00",
    "153lA00",
    # ... more structures
]

for struct_id in motif_containing_structures:
    src = cath_dir / struct_id
    dst = ref_dir / f"{struct_id}.pdb"
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {struct_id}")
```

### For Unconditional Generation Evaluation

For **unconditional generation**, you can use:

1. **Random CATH structures** (for structural similarity)
2. **Test set structures** (if you have a train/test split)
3. **Any high-quality protein structures** from PDB

---

## Finding Structures with Specific Motifs

### Method 1: Search CATH by Structure

If you know which CATH structures were used during training:

```bash
# Check your training logs or dataset
grep -r "cath" results/advanced_flow/advanced_flow_*/training.log

# Or check dataset cache
ls -la foldingdiff/cache_*.pkl
```

### Method 2: Extract from Training Dataset

```python
from foldingdiff.datasets import CathCanonicalAnglesDataset

# Load test set
test_dataset = CathCanonicalAnglesDataset(
    pdbs="cath",
    split="test",
    pad=128,
    min_length=40
)

# Get structure filenames
structure_files = test_dataset.fnames[:25]  # First 25 for example
print("Test set structures:", structure_files)
```

### Method 3: Use CATH Web Interface

1. Go to [CATH Database](http://www.cathdb.info/)
2. Search by domain classification
3. Download structures matching your criteria

---

## Quick Setup Script

Here's a script to quickly set up references from CATH:

```bash
#!/bin/bash
# setup_references.sh

REF_DIR="data/references"
CATH_DIR="data/cath/dompdb"
N_REFERENCES=25

mkdir -p "$REF_DIR"

# Option 1: Copy first N structures from CATH
echo "Copying first $N_REFERENCES structures from CATH..."
ls "$CATH_DIR" | head -$N_REFERENCES | while read f; do
    cp "$CATH_DIR/$f" "$REF_DIR/${f}.pdb"
done

# Option 2: Copy specific structures (modify list as needed)
# SPECIFIC_STRUCTURES=("12asA00" "152lA00" "153lA00")
# for struct in "${SPECIFIC_STRUCTURES[@]}"; do
#     if [ -f "$CATH_DIR/$struct" ]; then
#         cp "$CATH_DIR/$struct" "$REF_DIR/${struct}.pdb"
#     fi
# done

echo "Created $N_REFERENCES reference structures in $REF_DIR"
ls -lh "$REF_DIR" | head -10
```

Run it:
```bash
chmod +x setup_references.sh
./setup_references.sh
```

---

## Verification

After setting up references, verify they work:

```bash
# Check that references are found
python -c "
from pathlib import Path
from evaluations.evaluate_pipeline import find_pdb_files

ref_dir = Path('data/references')
refs = find_pdb_files(ref_dir)
print(f'Found {len(refs)} reference structures')
for r in refs[:5]:
    print(f'  - {r.name}')
"

# Test evaluation with references
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \
    --reference_dir data/references \
    --output_dir test_evaluation
```

---

## Common Issues and Solutions

### Issue 1: "No matching pairs found"

**Problem:** Generated and reference structures don't match by name or index.

**Solution:**
```bash
# Check generated structure names
ls results/.../two_motifs_short/pdb/ | head -5

# Check reference structure names
ls data/references/ | head -5

# Rename references to match (if needed)
cd data/references
for i in {0..24}; do
    old_name=$(ls | sed -n "$((i+1))p")
    new_name="sample_$(printf "%04d" $i).pdb"
    mv "$old_name" "$new_name"
done
```

### Issue 2: "Different number of structures"

**Problem:** You have 25 generated structures but only 10 references.

**Solution:** Either:
- Copy more references to match the number of generated structures
- Or the script will only evaluate the first N pairs (where N = min(generated, references))

### Issue 3: "Reference structures not found"

**Problem:** The `--reference_dir` path is incorrect.

**Solution:**
```bash
# Use absolute path
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir ... \
    --reference_dir /disk-10tb/flow_scaffold/data/references

# Or check path exists
ls -la data/references/
```

---

## Summary

1. **For motif scaffolding:** Use original structures that contain the motifs
2. **For unconditional generation:** Use any high-quality structures (CATH test set, PDB, etc.)
3. **For novelty evaluation:** Use the full CATH database (`data/cath/dompdb/`)
4. **File naming:** Match generated structure names for best results
5. **Directory structure:** Simple directory with PDB files is sufficient

**Quick Start:**
```bash
# 1. Create reference directory
mkdir -p data/references

# 2. Copy structures from CATH (or your source)
cp data/cath/dompdb/12asA00 data/references/sample_0000.pdb
# ... copy more as needed

# 3. Run evaluation
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \
    --reference_dir data/references
```

---

## Next Steps

After setting up references, you'll get **3 additional metric categories**:

1. ✅ **Structural Similarity** - RMSD, TM-score, GDT
2. ✅ **Motif Recovery** - Motif-specific metrics
3. ✅ **Sequence-Structure Compatibility** - Sequence alignment metrics

Check the results in:
- `{samples_dir}/{scenario}/evaluation/structural_similarity.csv`
- `{samples_dir}/{scenario}/evaluation/motif_recovery.csv`
- `{samples_dir}/{scenario}/evaluation/sequence_structure_compatibility.csv`

