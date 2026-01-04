#!/bin/bash
# PHASE 5: Model Comparison
# Usage: ./phase5_compare_advanced_flow.sh [EXPERIMENT_NAME]

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
    ANALYSIS_DIR="${OUTPUT_BASE}/analysis_${EXPERIMENT_NAME}"
fi

echo "=========================================="
echo "PHASE 5: MODEL COMPARISON"
echo "=========================================="
echo ""

echo "Comparing advanced model with baseline..."

# Run model comparison
python bin/compare_flow_models.py \
    --models enhanced advanced \
    --batch_size 4 \
    --seq_len 64 \
    --num_runs 5 \
    --output_file ${ANALYSIS_DIR}/model_comparison.json

echo ""
echo "✓ Model comparison complete!"
echo "Results saved to: ${ANALYSIS_DIR}/model_comparison.json"
echo ""
echo "To continue with report generation, run:"
echo "  ./phase6_report_advanced_flow.sh ${EXPERIMENT_NAME}"
echo ""

