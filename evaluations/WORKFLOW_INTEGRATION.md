# Evaluation Framework - Workflow Integration

## Overview

The comprehensive evaluation framework has been fully integrated into the training and sampling workflows. This document explains how evaluation fits into the complete pipeline.

## Workflow Structure

### Complete Pipeline Flow

```
1. TRAINING
   └─> Train flow matching model
       └─> Save checkpoints to {MODEL_DIR}

2. SAMPLING
   └─> Generate backbone structures
       └─> Save to {SAMPLES_DIR}/{scenario}/pdb/
       └─> Save metadata to {SAMPLES_DIR}/{scenario}/sampling_args.json

3. COMPREHENSIVE EVALUATION ⭐ NEW
   └─> Run evaluation framework
       └─> Compute all metrics (RMSD, TM-score, GDT, energy, diversity, etc.)
       └─> Save to {ANALYSIS_DIR}/evaluation/

4. ANALYSIS
   └─> Angle-based analysis (existing)
       └─> Ramachandran plots, diversity metrics
       └─> Save to {ANALYSIS_DIR}/

5. REPORT GENERATION
   └─> Combine evaluation + analysis results
       └─> Generate comprehensive report
```

## Integration Points

### 1. Training Scripts

**Files Modified:**
- `train_and_evaluate_enhanced_flow.sh`
- `train_and_evaluate_advanced_flow.sh`

**Changes:**
- Added **Phase 3: Comprehensive Evaluation** before analysis
- Calls `evaluations/evaluate_sampled_backbones.py` automatically
- Evaluation results included in final report

**Example:**
```bash
# Phase 3: Comprehensive Evaluation
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir ${SAMPLES_DIR} \
    --output_dir ${EVALUATION_DIR} \
    --n_threads $(nproc)
```

### 2. Sampling Scripts

**Files Modified:**
- `bin/sample_enhanced_flow.py`
- `bin/sample_advanced_flow.py`

**Changes:**
- Updated "Next steps" to point to comprehensive evaluation
- Removed references to old evaluation scripts
- Added guidance to use evaluation framework

**Before:**
```python
logger.info("2. Evaluate: python bin/evaluate_samples.py")
```

**After:**
```python
logger.info("2. Evaluate comprehensively: python evaluations/evaluate_sampled_backbones.py --samples_dir " + str(output_dir))
```

### 3. Evaluation Framework

**New Features:**
- Automatic detection of sampling output structure
- Extraction of motif regions from metadata
- Per-scenario and combined evaluation
- Integration with existing analysis pipeline

## Usage

### Automatic (Recommended)

The evaluation runs automatically as part of the training workflow:

```bash
# Run complete workflow (training + sampling + evaluation + analysis)
bash train_and_evaluate_enhanced_flow.sh
# or
bash train_and_evaluate_advanced_flow.sh
```

**What happens:**
1. Model trains → saves to `results/{model_type}/{experiment_name}/`
2. Samples generated → saves to `results/{model_type}/samples_{experiment_name}/`
3. **Evaluation runs automatically** → saves to `results/{model_type}/analysis_{experiment_name}/evaluation/`
4. Analysis runs → saves to `results/{model_type}/analysis_{experiment_name}/`
5. Report generated → includes both evaluation and analysis results

### Manual Evaluation

You can also run evaluation manually on existing samples:

```bash
# Evaluate all scenarios in a samples directory
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334

# With reference structures
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --reference_dir data/references

# With database for novelty
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir results/advanced_flow/samples_advanced_flow_251231_091334 \
    --database_dir data/cath/dompdb
```

## Output Structure

After running the complete workflow:

```
results/{model_type}/
├── {experiment_name}/              # Trained model
│   ├── models/
│   ├── logs/
│   └── config.json
│
├── samples_{experiment_name}/       # Generated samples
│   ├── short_single_motif/
│   │   ├── pdb/
│   │   ├── angles/
│   │   └── sampling_args.json
│   ├── medium_single_motif/
│   └── ...
│
└── analysis_{experiment_name}/     # Analysis results
    ├── evaluation/                 # ⭐ Comprehensive evaluation
    │   ├── short_single_motif/
    │   │   ├── structural_similarity.csv
    │   │   ├── motif_recovery.csv
    │   │   ├── energy_plausibility.csv
    │   │   ├── novelty_diversity.csv
    │   │   └── evaluation_summary.json
    │   ├── medium_single_motif/
    │   ├── ...
    │   └── combined_evaluation_summary.json
    │
    ├── metrics.json                # Angle-based analysis
    ├── comparison_table.csv
    ├── diversity_comparison.png
    └── EVALUATION_REPORT.md        # Combined report
```

## Evaluation Metrics

### Always Computed (No Reference Needed)

1. **Energy/Physical Plausibility**
   - Ramachandran plot quality
   - Van der Waals clashes
   - Overall quality scores
   - Rosetta energy (if available)
   - AlphaFold2 confidence (if available)

2. **Novelty & Diversity**
   - Structural diversity (pairwise RMSD/TM-score)
   - Sequence diversity
   - Novelty vs. database (if database provided)

### Requires Reference Structures

3. **Structural Similarity**
   - RMSD (Root Mean Square Deviation)
   - TM-score (Template Modeling Score)
   - GDT (Global Distance Test)

4. **Motif Recovery** (if motifs specified)
   - Motif RMSD
   - Motif TM-score
   - Motif preservation
   - Interface quality

5. **Sequence-Structure Compatibility**
   - Sequence recovery rates
   - Sequence similarity
   - Property-based recovery

## Benefits of Integration

### 1. Automatic Evaluation

- No manual steps required
- Runs immediately after sampling
- Consistent evaluation across all experiments

### 2. Comprehensive Metrics

- Goes beyond angle-based analysis
- Includes structural similarity, energy, diversity
- Academic-standard metrics (RMSD, TM-score, GDT)

### 3. Integrated Reports

- Evaluation results included in final reports
- Both angle-based and structure-based metrics
- Complete picture of model performance

### 4. Flexible Usage

- Can run standalone on existing samples
- Can integrate into custom workflows
- Supports reference structures and databases

## Comparison: Before vs After

### Before Integration

```
Training → Sampling → Basic Analysis → Report
                      (angles only)
```

**Limitations:**
- Only angle-based metrics
- No structural similarity
- No energy/plausibility
- No motif recovery metrics
- Manual evaluation required

### After Integration

```
Training → Sampling → Comprehensive Evaluation → Analysis → Report
                      (all metrics)            (angles)
```

**Benefits:**
- ✅ All evaluation metrics automatically computed
- ✅ Structural similarity (RMSD, TM-score, GDT)
- ✅ Energy/plausibility scores
- ✅ Motif recovery metrics
- ✅ Integrated into workflow

## Configuration

### Environment Variables

You can customize evaluation by setting environment variables:

```bash
# Optional: Reference structures for comparison
export REFERENCE_DIR="data/references"

# Optional: Database for novelty evaluation
export DATABASE_DIR="data/cath/dompdb"

# Optional: Enable Rosetta energy (requires Rosetta)
export COMPUTE_ROSETTA="true"

# Optional: Enable AlphaFold2 confidence (requires AF2)
export COMPUTE_ALPHAFOLD="true"
```

### In Training Scripts

The evaluation phase can be customized in the training scripts:

```bash
# In train_and_evaluate_*.sh
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir ${SAMPLES_DIR} \
    --output_dir ${EVALUATION_DIR} \
    ${REFERENCE_DIR:+--reference_dir "${REFERENCE_DIR}"} \
    ${DATABASE_DIR:+--database_dir "${DATABASE_DIR}"} \
    ${COMPUTE_ROSETTA:+--compute_rosetta} \
    ${COMPUTE_ALPHAFOLD:+--compute_alphafold} \
    --n_threads $(nproc)
```

## Troubleshooting

### Evaluation Not Running

If evaluation doesn't run automatically:
1. Check that `evaluations/evaluate_sampled_backbones.py` exists
2. Verify samples directory structure (should have `pdb/` subdirectories)
3. Check Python path and dependencies

### Missing Metrics

Some metrics require additional setup:
- **Rosetta energy**: Requires Rosetta installation
- **AlphaFold2 confidence**: Requires AlphaFold2 or pLDDT in B-factor column
- **Structural similarity**: Requires reference structures
- **Motif recovery**: Requires reference structures + motif indices

### Performance

For large datasets:
- Use `--n_threads` to control parallelism
- Evaluation runs per-scenario by default (better memory usage)
- Use `--evaluate_together` only if needed for combined diversity

## Next Steps

1. **Run complete workflow** to see evaluation in action
2. **Review evaluation results** in `{ANALYSIS_DIR}/evaluation/`
3. **Compare metrics** across different experiments
4. **Use results** to improve model training and sampling

## Summary

The evaluation framework is now fully integrated into the training and sampling workflows:

✅ **Automatic**: Runs as part of training workflow  
✅ **Comprehensive**: All metrics computed automatically  
✅ **Integrated**: Results included in final reports  
✅ **Flexible**: Can run standalone or as part of pipeline  
✅ **Academic-standard**: Uses established metrics (RMSD, TM-score, GDT)

The evaluation framework now seamlessly fits into the complete protein generation pipeline, providing comprehensive assessment of generated structures at every step.

