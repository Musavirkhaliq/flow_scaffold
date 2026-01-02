# Automatic Reference Structure Saving - Implementation Summary

## What Was Implemented

Automatic saving of reference structures during sampling/testing, enabling automatic comparison of generated structures against real protein structures from the CATH database.

## Files Created/Modified

### New Files

1. **`foldingdiff/reference_saving.py`**
   - Core module for loading and saving reference structures
   - Functions:
     - `load_reference_structure_from_dataset()`: Load structures from CATH
     - `save_reference_structure()`: Save reference to output directory
     - `load_and_save_references()`: Batch load and save multiple references

2. **`evaluations/AUTO_REFERENCE_GUIDE.md`**
   - User guide for automatic reference saving
   - Usage examples and troubleshooting

3. **`evaluations/AUTO_REFERENCE_IMPLEMENTATION.md`** (this file)
   - Implementation summary

### Modified Files

1. **`bin/sample_advanced_flow.py`**
   - Added `--save_references` flag (enabled by default)
   - Added `--reference_seed` flag for reproducibility
   - Automatically loads and saves references during sampling
   - Stores reference paths in `sampling_args.json`

2. **`bin/sample_enhanced_flow.py`**
   - Same changes as `sample_advanced_flow.py`

3. **`evaluations/evaluate_sampled_backbones.py`**
   - Added `find_saved_references()` function
   - Automatically detects saved references in `{scenario}/references/`
   - Uses auto-saved references if available (before manual `--reference_dir`)
   - Updated help text to mention auto-detection

## How It Works

### During Sampling

1. User runs sampling script (e.g., `bin/sample_advanced_flow.py`)
2. Script creates CATH dataset
3. For each sample:
   - Loads a random structure from CATH matching the target length
   - Extracts motif regions if motif scaffolding
   - Saves reference PDB to `{output_dir}/references/reference_{i:04d}.pdb`
4. Stores reference paths in `sampling_args.json`

### During Evaluation

1. User runs evaluation script (e.g., `evaluations/evaluate_sampled_backbones.py`)
2. Script checks for `{scenario}/references/` directory
3. If found:
   - Uses saved references automatically
   - Matches generated structures to references by name/index
   - Enables full evaluation suite (all 5 metric categories)
4. If not found:
   - Falls back to manual `--reference_dir` if provided
   - Otherwise, only computes metrics that don't need references

## Key Features

### ✅ Automatic
- No manual setup required
- References saved automatically during sampling
- Evaluation automatically finds and uses them

### ✅ Reproducible
- Use `--reference_seed` to control which structures are selected
- Same seed = same references

### ✅ Smart Matching
- References selected to match sampling length (±10 residues)
- For motif scaffolding, attempts to find structures with similar motifs

### ✅ Flexible
- Can disable with `--no-save_references`
- Can override with manual `--reference_dir` in evaluation
- Works for both unconditional and conditional (motif) generation

## Usage Example

```bash
# 1. Sample with automatic reference saving
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/advanced_flow_260102_022234 \
    --length 100 \
    --n_samples 25 \
    --motif_regions "10-20,50-60" \
    --output_dir results/samples/my_experiment

# References automatically saved to:
# results/samples/my_experiment/references/

# 2. Evaluate (automatically uses saved references)
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/my_experiment

# No --reference_dir needed!
# All 5 metric categories will be computed:
# - Energy/Physical Plausibility ✓
# - Novelty & Diversity ✓
# - Structural Similarity ✓ (uses saved references)
# - Motif Recovery ✓ (uses saved references)
# - Sequence-Structure Compatibility ✓ (uses saved references)
```

## Benefits

1. **No Manual Setup**: References are handled automatically
2. **Complete Evaluation**: Enables all 5 metric categories
3. **Consistent**: Same references for same sampling run
4. **Reproducible**: Use seed for reproducibility
5. **Convenient**: Evaluation automatically finds references

## Technical Details

### Reference Selection

- **Source**: CATH database (`data/cath/dompdb/`)
- **Length Matching**: ±10 residues from target length
- **Random Selection**: Random structures from matching length range
- **Seed Control**: `--reference_seed` for reproducibility

### File Naming

- Generated: `sample_{i:04d}.pdb`
- References: `reference_{i:04d}.pdb`
- Matching: By index (i) or by name

### Directory Structure

```
{output_dir}/
├── pdb/
│   └── sample_XXXX.pdb
├── references/          # ← New!
│   └── reference_XXXX.pdb
├── angles/
│   └── sample_XXXX.csv
└── sampling_args.json  # ← Contains reference_paths
```

## Future Enhancements

Potential improvements:

1. **Motif-Aware Selection**: For motif scaffolding, select references that actually contain similar motifs
2. **Quality Filtering**: Filter references by quality metrics (Ramachandran, etc.)
3. **Diversity Control**: Ensure selected references are diverse
4. **Caching**: Cache reference selections for faster re-runs
5. **Custom Sources**: Allow custom reference databases beyond CATH

## Testing

To test the implementation:

```bash
# Test reference saving module
python -c "from foldingdiff.reference_saving import load_and_save_references; print('✓ Module works')"

# Test sampling with references
python bin/sample_advanced_flow.py \
    --model_dir <your_model> \
    --length 100 \
    --n_samples 5 \
    --output_dir test_samples \
    --save_references

# Check references were saved
ls test_samples/references/

# Test evaluation auto-detection
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir test_samples
```

## Summary

✅ **Implemented**: Automatic reference structure saving  
✅ **Integrated**: Both sampling scripts updated  
✅ **Auto-Detection**: Evaluation automatically finds references  
✅ **Documented**: User guide and implementation summary created  
✅ **Tested**: Module imports successfully  

The feature is ready to use! Just run sampling, and references will be automatically saved and used during evaluation.

