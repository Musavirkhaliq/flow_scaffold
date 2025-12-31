## Benchmarks Overview

This folder contains utilities to **compare the enhanced flow-scaffolding model** against
standard baselines (e.g. **FoldingDiff**, **motif benchmarks**, or any other method you
have metrics for).

The goal is to answer: **“How much better is our model, across all key metrics?”**

### Layout

- `benchmarks/README.md` – this file
- `benchmarks/baselines/` – reference metrics for external baselines
  - `foldingdiff_baseline.json` – placeholder for FoldingDiff metrics
  - `motif_benchmark.json` – placeholder for motif-scaffolding benchmarks
- `benchmarks/compare_to_baselines.py` – script to compare our results to baselines

### What we compare

The comparator is designed to use the metrics already produced by the pipeline:

- Backbone / angle metrics (from `metrics.json`):
  - Mean pairwise diversity
  - Ramachandran alpha / beta fractions
  - Angle means / stds
- Validity metrics:
  - NaN / Inf presence
  - Out-of-range angle fraction
- Any additional high-level scores you put in the baseline JSON (e.g. TM-score, RMSD)

### How to generate our metrics

Run the full training + sampling + analysis pipeline (already set up):

```bash
bash train_and_evaluate_enhanced_flow.sh
```

This will create (paths will have a timestamp suffix):

- `results/enhanced_flow/enhanced_flow_*/` – trained model
- `results/enhanced_flow/samples_enhanced_flow_*/` – sampled angles/pdbs
- `results/enhanced_flow/analysis_enhanced_flow_*/metrics.json` – per-scenario metrics
- `results/enhanced_flow/analysis_enhanced_flow_*/comparison_table.csv` – summary table

### How to define baselines

Fill in the JSON files under `benchmarks/baselines/` with metrics from:

- Published FoldingDiff results
- Motif-benchmark papers / internal baselines
- Other internal models

You can add more baseline JSONs with the same schema if needed.

Example (see actual file for full structure):

```json
{
  "name": "foldingdiff_paper",
  "source": "Paper XYZ",
  "metrics": {
    "tm_score_mean": 0.70,
    "rmsd_mean": 3.2
  }
}
```

### How to run comparison

After you have:

1. Our analysis directory (e.g. `results/enhanced_flow/analysis_enhanced_flow_251230_102423`)
2. Baseline JSON files filled in under `benchmarks/baselines/`

Run:

```bash
cd flow_scaf
python benchmarks/compare_to_baselines.py \
  --analysis_dir results/enhanced_flow/analysis_enhanced_flow_251230_102423 \
  --baseline foldingdiff_baseline \
  --baseline motif_benchmark
```

This will print:

- A textual comparison table for each scenario (unconditional, motif settings, etc.)
- A high-level summary (e.g. “Our mean diversity is X% higher than FoldingDiff”)

### Notes

- This benchmark layer is **read-only** with respect to your results; it does not re-run
  training or sampling, it just consumes existing outputs.
- You can extend it to:
  - Read TM-score / RMSD from `validate_structures.py` outputs
  - Compare sequence-level metrics (entropy, repetition) from `analyze_sequence_quality.py`


