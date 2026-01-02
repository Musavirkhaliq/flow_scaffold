# Quick Start - Evaluation Script

## Simple Usage

Just provide the samples directory path:

```bash
python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234
```

That's it! The script will:
- ✅ Automatically find all scenarios in the directory
- ✅ Extract motif regions from metadata
- ✅ Evaluate each scenario separately
- ✅ Generate comprehensive metrics
- ✅ Save results to `{samples_dir}/evaluation/`

## Output

For each scenario, you'll get:

```
{samples_dir}/{scenario}/evaluation/
  ├── energy_plausibility.csv          # Ramachandran, VDW clashes, quality
  ├── novelty_diversity.csv            # Structural diversity, novelty
  ├── structural_similarity.csv          # RMSD, TM-score, GDT (if reference provided)
  ├── motif_recovery.csv               # Motif metrics (if motifs specified)
  ├── sequence_structure_compatibility.csv  # Sequence metrics (if reference provided)
  └── evaluation_summary.json           # Overall statistics
```

## Examples

### Basic Evaluation
```bash
python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234
```

### With Reference Structures
```bash
python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234 \
    --reference_dir data/references
```

### With Database for Novelty
```bash
python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234 \
    --database_dir data/cath/dompdb
```

### Full Options
```bash
python evaluations/run_evaluation.py results/advanced_flow/samples_advanced_flow_260102_022234 \
    --reference_dir data/references \
    --database_dir data/cath/dompdb \
    --compute_rosetta \
    --n_threads 8
```

## What Gets Evaluated

### Always Computed (No Reference Needed)
- ✅ **Energy/Physical Plausibility**: Ramachandran quality, VDW clashes, overall quality
- ✅ **Novelty & Diversity**: Structural diversity, sequence diversity

### Requires Reference Structures
- ✅ **Structural Similarity**: RMSD, TM-score, GDT
- ✅ **Sequence-Structure Compatibility**: Recovery rates, similarity

### Requires Reference + Motif Indices
- ✅ **Motif Recovery**: Motif preservation, interface quality

## Notes

- **TMalign**: If not installed, TM-score metrics will show warnings but evaluation continues
- **Rosetta**: Optional, use `--compute_rosetta` if available
- **AlphaFold2**: Optional, use `--compute_alphafold` if available
- **Parallel Processing**: Uses all CPU cores by default (use `--n_threads` to control)

## Quick Check

After running, check the summary:
```bash
cat results/advanced_flow/samples_advanced_flow_260102_022234/unconditional_short/evaluation/evaluation_summary.json
```

