# Advanced Flow Matching - Phase Scripts

This directory contains separate scripts for each phase of the Advanced Flow Matching workflow. You can run any phase independently, or run them sequentially.

## Scripts Overview

1. **`config_advanced_flow.sh`** - Shared configuration file (sourced by all phase scripts)
2. **`phase1_train_advanced_flow.sh`** - Train the advanced flow matching model
3. **`phase2_sample_advanced_flow.sh`** - Generate samples from the trained model
4. **`phase3_evaluate_advanced_flow.sh`** - Comprehensive evaluation of samples
5. **`phase4_analyze_advanced_flow.sh`** - Detailed analysis and visualization
6. **`phase5_compare_advanced_flow.sh`** - Model comparison with baselines
7. **`phase6_report_advanced_flow.sh`** - Generate comprehensive markdown report

## Usage

### Running Individual Phases

Each phase script can be run independently. You can pass either:
- **Experiment name only**: `advanced_flow_260103_112903`
- **Full path**: `/disk-10tb/flow_scaffold/results/advanced_flow/advanced_flow_260103_112903`

The scripts will automatically extract the experiment name from full paths.

```bash
# Phase 1: Train model
./phase1_train_advanced_flow.sh [EXPERIMENT_NAME]

# Phase 2: Sample from model
./phase2_sample_advanced_flow.sh [EXPERIMENT_NAME]
# Or with full path:
./phase2_sample_advanced_flow.sh /disk-10tb/flow_scaffold/results/advanced_flow/advanced_flow_260103_112903

# Phase 3: Evaluate samples
./phase3_evaluate_advanced_flow.sh [EXPERIMENT_NAME]

# Phase 4: Analyze results
./phase4_analyze_advanced_flow.sh [EXPERIMENT_NAME]

# Phase 5: Compare models
./phase5_compare_advanced_flow.sh [EXPERIMENT_NAME]

# Phase 6: Generate report
./phase6_report_advanced_flow.sh [EXPERIMENT_NAME]
```

### Running All Phases Sequentially

You can run all phases in order:

```bash
EXPERIMENT_NAME="my_experiment_$(date +%y%m%d_%H%M%S)"

./phase1_train_advanced_flow.sh ${EXPERIMENT_NAME}
./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME}
./phase3_evaluate_advanced_flow.sh ${EXPERIMENT_NAME}
./phase4_analyze_advanced_flow.sh ${EXPERIMENT_NAME}
./phase5_compare_advanced_flow.sh ${EXPERIMENT_NAME}
./phase6_report_advanced_flow.sh ${EXPERIMENT_NAME}
```

### Using an Existing Experiment

If you want to run phases 2-6 on an already-trained model:

```bash
EXPERIMENT_NAME="advanced_flow_260103_112903"  # Use your existing experiment name

./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME}
./phase3_evaluate_advanced_flow.sh ${EXPERIMENT_NAME}
./phase4_analyze_advanced_flow.sh ${EXPERIMENT_NAME}
./phase5_compare_advanced_flow.sh ${EXPERIMENT_NAME}
./phase6_report_advanced_flow.sh ${EXPERIMENT_NAME}
```

## Configuration

All scripts source `config_advanced_flow.sh` which contains:

- Training parameters (epochs, batch size, learning rate, etc.)
- Model architecture (hidden size, layers, heads)
- Advanced features (sequence augmentation, geometric inverse design, etc.)
- Sampling parameters (steps, guidance scale, etc.)
- Dataset configuration (CATH, AlphaFold directories)

You can modify `config_advanced_flow.sh` to change parameters for all phases, or override specific variables in individual scripts.

## Phase Dependencies

- **Phase 1** (Training): No dependencies
- **Phase 2** (Sampling): Requires Phase 1 (trained model)
- **Phase 3** (Evaluation): Requires Phase 2 (samples)
- **Phase 4** (Analysis): Requires Phase 2 (samples)
- **Phase 5** (Comparison): Can run independently (compares models)
- **Phase 6** (Report): Requires Phase 4 (analysis results), optionally Phase 3 and 5

## Output Structure

All outputs are organized under `results/advanced_flow/`:

```
results/advanced_flow/
├── ${EXPERIMENT_NAME}/              # Trained model (Phase 1)
│   ├── models/
│   │   ├── best_by_train/
│   │   └── best_by_valid/
│   └── logs/
├── samples_${EXPERIMENT_NAME}/       # Generated samples (Phase 2)
│   └── [scenario directories]/
└── analysis_${EXPERIMENT_NAME}/     # Analysis results (Phases 3-6)
    ├── evaluation/                   # Phase 3 results
    ├── advanced_metrics.json         # Phase 4 results
    ├── advanced_comparison_table.csv # Phase 4 results
    ├── model_comparison.json        # Phase 5 results
    └── ADVANCED_EVALUATION_REPORT.md # Phase 6 report
```

## Examples

### Example 1: Train and Sample Only

```bash
EXPERIMENT_NAME="quick_test"
./phase1_train_advanced_flow.sh ${EXPERIMENT_NAME}
./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME}
```

### Example 2: Re-analyze Existing Samples

```bash
EXPERIMENT_NAME="advanced_flow_260103_112903"
./phase4_analyze_advanced_flow.sh ${EXPERIMENT_NAME}
./phase6_report_advanced_flow.sh ${EXPERIMENT_NAME}
```

### Example 3: Full Workflow

```bash
EXPERIMENT_NAME="full_experiment_$(date +%y%m%d_%H%M%S)"
./phase1_train_advanced_flow.sh ${EXPERIMENT_NAME} && \
./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME} && \
./phase3_evaluate_advanced_flow.sh ${EXPERIMENT_NAME} && \
./phase4_analyze_advanced_flow.sh ${EXPERIMENT_NAME} && \
./phase5_compare_advanced_flow.sh ${EXPERIMENT_NAME} && \
./phase6_report_advanced_flow.sh ${EXPERIMENT_NAME}
```

## Notes

- Each script checks for required dependencies (e.g., Phase 2 checks if model exists)
- Experiment names are optional - if not provided, a timestamp-based name is generated
- All scripts use the same configuration from `config_advanced_flow.sh`
- Scripts are designed to be idempotent (safe to re-run)

## Troubleshooting

If a phase fails:
1. Check the error message - it will indicate what's missing
2. Ensure previous phases completed successfully
3. Verify the experiment name matches across phases
4. Check that required directories and files exist

For example, if Phase 2 fails with "Model directory not found", ensure Phase 1 completed successfully and the experiment name is correct.

