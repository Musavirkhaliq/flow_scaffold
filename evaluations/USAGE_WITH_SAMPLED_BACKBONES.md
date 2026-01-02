# Using Evaluation Framework with Sampled Backbones

This guide shows how to use the evaluation framework with backbones sampled from the flow matching model.

## Standard Sampling Output Structure

When you sample backbones using the flow matching pipeline, the output structure is:

```
{samples_dir}/
  {scenario_name}/
    pdb/
      sample_0000.pdb
      sample_0001.pdb
      ...
    angles/
      sample_0000.csv
      sample_0001.csv
      ...
    sampling_args.json  # Contains motif regions and other metadata
    statistics.json     # Alternative metadata file
```

## Quick Start

### Option 1: Automatic Evaluation (Recommended)

Use the convenience script that automatically detects scenarios and extracts motif regions:

```bash
# Evaluate all scenarios in a samples directory
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334

# Evaluate specific scenarios
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --scenarios short_single_motif medium_single_motif

# With reference structures for comparison
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --reference_dir data/references

# With database for novelty evaluation
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --database_dir data/cath/dompdb
```

### Option 2: Direct Evaluation

Use the main evaluation pipeline by pointing directly to the `pdb/` subdirectory:

```bash
# Evaluate a single scenario
python evaluations/evaluate_pipeline.py \
    --generated_dir results/advanced_flow/samples_advanced_flow_251231_091334/short_single_motif/pdb \
    --output_dir results/advanced_flow/samples_advanced_flow_251231_091334/short_single_motif/evaluation

# With motif indices (if not in metadata)
python evaluations/evaluate_pipeline.py \
    --generated_dir results/advanced_flow/samples_advanced_flow_251231_091334/short_single_motif/pdb \
    --motif_indices "10-20" \
    --output_dir results/advanced_flow/samples_advanced_flow_251231_091334/short_single_motif/evaluation
```

## Features

### Automatic Motif Region Detection

The `evaluate_sampled_backbones.py` script automatically extracts motif regions from:
- `sampling_args.json`
- `statistics.json`
- `advanced_summary.json`

Example metadata format:
```json
{
  "motif_regions": "10-20,30-40",
  "length": 100,
  "guidance_scale": 2.0
}
```

### Per-Scenario Evaluation

By default, each scenario is evaluated separately, creating:
- `{scenario}/evaluation/` - Individual scenario results
- `evaluation_combined/` - Combined summary across all scenarios

### Evaluation Metrics

For each scenario, the following metrics are computed:

1. **Energy/Physical Plausibility** (no reference needed)
   - Ramachandran plot quality
   - Van der Waals clashes
   - Rosetta energy (if available)
   - AlphaFold2 confidence (if available)

2. **Novelty & Diversity** (no reference needed)
   - Structural diversity (pairwise RMSD/TM-score)
   - Sequence diversity
   - Novelty vs. database (if database provided)

3. **Structural Similarity** (requires reference)
   - RMSD
   - TM-score
   - GDT scores

4. **Motif Recovery** (requires reference + motif indices)
   - Motif RMSD
   - Motif TM-score
   - Interface quality

5. **Sequence-Structure Compatibility** (requires reference)
   - Sequence recovery rates
   - Sequence similarity

## Output Files

Each evaluation creates:

```
{evaluation_dir}/
  structural_similarity.csv          # RMSD, TM-score, GDT
  motif_recovery.csv                 # Motif-specific metrics
  sequence_structure_compatibility.csv # Sequence metrics
  energy_plausibility.csv            # Quality/plausibility
  novelty_diversity.csv              # Diversity/novelty
  evaluation_summary.json            # Overall statistics
```

## Examples

### Example 1: Evaluate All Scenarios

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --output_dir evaluation_results
```

This will:
- Find all scenarios (short_single_motif, medium_single_motif, etc.)
- Extract motif regions from metadata
- Evaluate each scenario separately
- Create combined summary

### Example 2: Evaluate with Reference Structures

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --reference_dir data/references \
    --output_dir evaluation_results
```

### Example 3: Evaluate Specific Scenarios

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --scenarios short_single_motif two_motifs_short \
    --output_dir evaluation_results
```

### Example 4: Full Evaluation with All Options

```bash
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --reference_dir data/references \
    --database_dir data/cath/dompdb \
    --compute_rosetta \
    --n_threads 8 \
    --output_dir evaluation_results
```

## Integration with Sampling Pipeline

You can integrate evaluation directly into your sampling workflow:

```bash
# After sampling
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/my_model \
    --length 100 \
    --n_samples 25 \
    --output_dir results/samples/my_scenario \
    --save_pdb

# Immediately evaluate
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/samples/my_scenario
```

## Reading Results

### Python

```python
import pandas as pd
import json

# Load evaluation results
energy_df = pd.read_csv("evaluation_results/energy_plausibility.csv")
struct_df = pd.read_csv("evaluation_results/structural_similarity.csv")

# Load summary
with open("evaluation_results/evaluation_summary.json") as f:
    summary = json.load(f)

print(f"Mean RMSD: {summary['mean_RMSD']:.2f} Å")
print(f"Mean TM-score: {summary['mean_TM_score']:.3f}")
```

### Command Line

```bash
# Quick summary
cat evaluation_results/evaluation_summary.json | jq

# Check energy scores
head -5 evaluation_results/energy_plausibility.csv

# Check structural similarity
head -5 evaluation_results/structural_similarity.csv
```

## Troubleshooting

### No PDB files found

If you get "No generated structures found", check:
1. The directory structure (should have `pdb/` subdirectory)
2. File naming (should be `sample_XXXX.pdb`)
3. Use `--generated_dir` pointing directly to `pdb/` subdirectory

### Motif indices not found

If motif regions aren't automatically detected:
1. Check metadata files exist (`sampling_args.json`, etc.)
2. Manually specify with `--motif_indices "10-20,30-40"`
3. Create a JSON file with motif indices list

### Performance

For large datasets:
- Use `--n_threads` to control parallelism
- Evaluate scenarios separately (default) for better memory usage
- Use `--evaluate_together` only if you need combined diversity metrics

## Next Steps

After evaluation:
1. **Analyze results** - Check CSV files for detailed metrics
2. **Compare scenarios** - Use combined summary to compare different sampling configurations
3. **Visualize** - Use PyMOL or other tools to visualize structures
4. **Iterate** - Use results to improve sampling parameters

