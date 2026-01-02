# Evaluation Framework Integration Summary

## Overview

The evaluation framework is now fully integrated with the sampled backbone pipeline. Here's how everything works together.

## File Structure

```
evaluations/
├── __init__.py                          # Package exports
├── README.md                            # Main documentation
├── USAGE_WITH_SAMPLED_BACKBONES.md      # This guide
├── INTEGRATION_SUMMARY.md               # This file
├── evaluate_pipeline.py                 # Main evaluation pipeline
├── evaluate_sampled_backbones.py        # Convenience script for sampled backbones
├── structural_similarity.py            # RMSD, TM-score, GDT
├── motif_recovery.py                    # Motif-specific metrics
├── sequence_structure_compatibility.py  # Sequence metrics
├── energy_plausibility.py               # Quality/plausibility
└── novelty_diversity.py                 # Diversity/novelty
```

## How It Works

### 1. Sampling Pipeline Output

When you run the sampling pipeline:

```bash
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/my_model \
    --length 100 \
    --motif_regions "10-20,30-40" \
    --output_dir results/samples/my_scenario \
    --save_pdb
```

**Output structure:**
```
results/samples/my_scenario/
  pdb/
    sample_0000.pdb
    sample_0001.pdb
    ...
  angles/
    sample_0000.csv
    ...
  sampling_args.json  # Contains: {"motif_regions": "10-20,30-40", ...}
```

### 2. Evaluation Pipeline

The evaluation framework automatically:
- ✅ Detects the `pdb/` subdirectory structure
- ✅ Extracts motif regions from `sampling_args.json`
- ✅ Finds all PDB files
- ✅ Computes all relevant metrics

**Usage:**
```bash
# Automatic evaluation (recommended)
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/my_scenario

# Or use main pipeline directly
python evaluations/evaluate_pipeline.py \
    --generated_dir results/samples/my_scenario/pdb \
    --motif_indices "10-20,30-40"
```

### 3. Evaluation Output

**Generated files:**
```
results/samples/my_scenario/evaluation/
  structural_similarity.csv          # If reference provided
  motif_recovery.csv                 # If reference + motif indices
  sequence_structure_compatibility.csv # If reference provided
  energy_plausibility.csv            # Always generated
  novelty_diversity.csv              # Always generated
  evaluation_summary.json            # Summary statistics
```

## Key Features

### ✅ Automatic Detection

- **PDB files**: Automatically finds files in `pdb/` subdirectory
- **Motif regions**: Extracts from `sampling_args.json`, `statistics.json`, or `advanced_summary.json`
- **Scenarios**: Detects multiple scenarios in a samples directory

### ✅ Flexible Evaluation

- **Per-scenario**: Evaluate each scenario separately (default)
- **Combined**: Evaluate all scenarios together
- **Selective**: Evaluate specific scenarios only

### ✅ Comprehensive Metrics

1. **Energy/Plausibility** (no reference needed)
   - Ramachandran quality
   - VDW clashes
   - Rosetta energy (optional)
   - AlphaFold2 confidence (optional)

2. **Novelty/Diversity** (no reference needed)
   - Structural diversity
   - Sequence diversity
   - Novelty vs. database (if provided)

3. **Structural Similarity** (requires reference)
   - RMSD, TM-score, GDT

4. **Motif Recovery** (requires reference + motifs)
   - Motif preservation
   - Interface quality

5. **Sequence-Structure** (requires reference)
   - Recovery rates
   - Similarity metrics

## Example Workflow

### Complete Pipeline

```bash
# 1. Sample backbones
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/my_model \
    --length 100 \
    --motif_regions "10-20,30-40" \
    --n_samples 25 \
    --output_dir results/samples/motif_scaffold \
    --save_pdb

# 2. Evaluate (automatic)
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/motif_scaffold

# 3. Check results
cat results/samples/motif_scaffold/evaluation/evaluation_summary.json
```

### With Reference Structures

```bash
# Evaluate with reference for comparison
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/motif_scaffold \
    --reference_dir data/references
```

### With Database for Novelty

```bash
# Evaluate novelty against CATH database
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/motif_scaffold \
    --database_dir data/cath/dompdb
```

## Integration Points

### 1. Sampling Scripts

The sampling scripts (`bin/sample_*.py`) save:
- PDB files in `{output_dir}/pdb/`
- Metadata in `sampling_args.json` with `motif_regions` field

### 2. Evaluation Scripts

The evaluation scripts:
- Read from `pdb/` subdirectory automatically
- Extract `motif_regions` from metadata
- Save results in `evaluation/` subdirectory

### 3. Pipeline Scripts

You can integrate evaluation into pipeline scripts:

```bash
# In your pipeline script
python bin/sample_advanced_flow.py ... --output_dir ${OUTPUT_DIR}

# Immediately evaluate
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir ${OUTPUT_DIR}
```

## Metadata Format

The evaluation framework expects metadata in this format:

```json
{
  "motif_regions": "10-20,30-40",  // String format
  "length": 100,
  "guidance_scale": 2.0,
  ...
}
```

Or alternatively:
```json
{
  "motif_regions_str": "10-20,30-40",
  "motif_indices": [10, 11, 12, ..., 20, 30, 31, ..., 40]
}
```

## Benefits

1. **Automatic**: No need to manually specify motif regions
2. **Consistent**: Uses same structure as sampling pipeline
3. **Comprehensive**: All metrics computed automatically
4. **Flexible**: Works with or without references
5. **Scalable**: Handles multiple scenarios efficiently

## Next Steps

1. **Run evaluation** on your sampled backbones
2. **Analyze results** using the CSV files
3. **Compare scenarios** using the combined summary
4. **Iterate** on sampling parameters based on results

## Support

For more details, see:
- `README.md` - Full documentation
- `USAGE_WITH_SAMPLED_BACKBONES.md` - Usage examples
- `evaluate_pipeline.py --help` - Command-line options

