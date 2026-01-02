#!/bin/bash
# Integration script to add comprehensive evaluation to training/sampling workflows
# This can be called from train_and_evaluate_*.sh scripts

set -e

SAMPLES_DIR="${1}"
EVALUATION_DIR="${2:-${SAMPLES_DIR}/evaluation}"
REFERENCE_DIR="${3:-}"
DATABASE_DIR="${4:-}"
COMPUTE_ROSETTA="${5:-false}"
COMPUTE_ALPHAFOLD="${6:-false}"

if [ -z "${SAMPLES_DIR}" ]; then
    echo "Usage: $0 <samples_dir> [evaluation_dir] [reference_dir] [database_dir] [compute_rosetta] [compute_alphafold]"
    exit 1
fi

echo "=========================================="
echo "COMPREHENSIVE EVALUATION"
echo "=========================================="
echo ""
echo "Samples directory: ${SAMPLES_DIR}"
echo "Evaluation output: ${EVALUATION_DIR}"
echo ""

# Check if samples directory exists
if [ ! -d "${SAMPLES_DIR}" ]; then
    echo "Error: Samples directory not found: ${SAMPLES_DIR}"
    exit 1
fi

# Run comprehensive evaluation
python evaluations/evaluate_sampled_backbones.py \
    --samples_dir "${SAMPLES_DIR}" \
    --output_dir "${EVALUATION_DIR}" \
    ${REFERENCE_DIR:+--reference_dir "${REFERENCE_DIR}"} \
    ${DATABASE_DIR:+--database_dir "${DATABASE_DIR}"} \
    ${COMPUTE_ROSETTA:+--compute_rosetta} \
    ${COMPUTE_ALPHAFOLD:+--compute_alphafold} \
    --n_threads $(nproc)

echo ""
echo "✓ Comprehensive evaluation complete!"
echo "  Results: ${EVALUATION_DIR}"
echo ""

