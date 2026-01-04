#!/bin/bash
# PHASE 3: Comprehensive Evaluation
# Usage: ./phase3_evaluate_advanced_flow.sh [EXPERIMENT_NAME]

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source shared configuration
source config_advanced_flow.sh

# Function to extract experiment name from path or name
extract_experiment_name() {
    local input="$1"
    if [ -z "$input" ]; then
        return
    fi
    # If it's a full path, extract just the last component
    if [[ "$input" == *"/"* ]]; then
        basename "$input"
    else
        echo "$input"
    fi
}

# Allow experiment name override
if [ -n "$1" ]; then
    EXPERIMENT_NAME="$(extract_experiment_name "$1")"
    SAMPLES_DIR="${OUTPUT_BASE}/samples_${EXPERIMENT_NAME}"
    ANALYSIS_DIR="${OUTPUT_BASE}/analysis_${EXPERIMENT_NAME}"
fi

# Check if samples exist
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "Error: Samples directory not found: ${SAMPLES_DIR}"
    echo "Please generate samples first using: ./phase2_sample_advanced_flow.sh"
    exit 1
fi

# Check if any scenario directories with pdb/ subdirectories exist
SCENARIOS_WITH_PDB=$(find "$SAMPLES_DIR" -type d -name "pdb" 2>/dev/null | wc -l)

if [ "$SCENARIOS_WITH_PDB" -eq 0 ]; then
    echo "Error: No sampled PDB files found in ${SAMPLES_DIR}"
    echo ""
    echo "The evaluation script requires scenario directories with 'pdb/' subdirectories."
    echo "Found scenario directories:"
    find "$SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read dir; do
        echo "  - $(basename "$dir")"
        if [ -d "$dir/references" ]; then
            echo "    (contains references but no pdb/ directory)"
        fi
    done
    echo ""
    echo "This usually means:"
    echo "  1. Sampling didn't complete successfully, or"
    echo "  2. PDB files weren't saved (check if --save_pdb was used)"
    echo ""
    echo "Please run sampling again:"
    echo "  ./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME}"
    exit 1
fi

echo "=========================================="
echo "PHASE 3: COMPREHENSIVE EVALUATION"
echo "=========================================="
echo ""

# Run comprehensive evaluation using the evaluation framework
EVALUATION_DIR="${ANALYSIS_DIR}/evaluation"

echo "Running comprehensive evaluation..."
echo "  Samples: ${SAMPLES_DIR}"
echo "  Output: ${EVALUATION_DIR}"
echo "  Found ${SCENARIOS_WITH_PDB} scenario(s) with PDB files"
echo ""

python evaluations/evaluate_sampled_backbones.py \
    --samples_dir ${SAMPLES_DIR} \
    --output_dir ${EVALUATION_DIR} \
    --n_threads $(nproc)

echo ""
echo "✓ Comprehensive evaluation complete!"
echo "Results saved to: ${EVALUATION_DIR}"
echo ""
echo "To continue with analysis, run:"
echo "  ./phase4_analyze_advanced_flow.sh ${EXPERIMENT_NAME}"
echo ""

