#!/bin/bash
# Standalone script to run OmegaFold structure prediction on designed sequences

set -e

echo "================================================================================"
echo "OMEGAFOLD STRUCTURE PREDICTION - STANDALONE"
echo "================================================================================"
echo ""

# Default parameters
INPUT_DIR=""
OUTPUT_DIR="predictions"
GPUS="0"
BATCH_SIZE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_omegafold.sh --input_dir <fasta_dir> [options]"
            echo ""
            echo "Required:"
            echo "  --input_dir DIR          Directory containing FASTA files"
            echo ""
            echo "Optional:"
            echo "  --output_dir DIR         Output directory (default: predictions)"
            echo "  --gpus IDS               GPU IDs to use, space-separated (default: 0)"
            echo "  --batch_size N           Batch size (default: 1)"
            echo ""
            echo "Example:"
            echo "  bash run_omegafold.sh --input_dir sequences --gpus \"0 1\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check for FASTA files
FASTA_COUNT=$(find "$INPUT_DIR" -name "*.fa" -o -name "*.fasta" | wc -l)
if [ "$FASTA_COUNT" -eq 0 ]; then
    echo "Error: No FASTA files found in $INPUT_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Input directory:  $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  FASTA files:      $FASTA_COUNT"
echo "  GPUs:             $GPUS"
echo "  Batch size:       $BATCH_SIZE"
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "✗ Python not found"
    exit 1
fi
echo "✓ Python found"

# Check if omegafold conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "omegafold" ]; then
    echo "⚠ OmegaFold conda environment not activated"
    echo ""
    echo "Please activate the omegafold environment first:"
    echo "  conda activate omegafold"
    echo ""
    echo "If not installed, create it with:"
    echo "  conda create -n omegafold python=3.9"
    echo "  conda activate omegafold"
    echo "  pip install omegafold"
    echo ""
    exit 1
fi
echo "✓ OmegaFold environment activated: $CONDA_DEFAULT_ENV"

# Check OmegaFold command
if ! command -v omegafold &> /dev/null; then
    echo "✗ OmegaFold command not found"
    echo ""
    echo "Install OmegaFold in the current environment:"
    echo "  pip install omegafold"
    exit 1
fi
echo "✓ OmegaFold command found"

# Check if we have the multi-GPU script
if [ -f "bin/omegafold_across_gpus.py" ]; then
    USE_MULTI_GPU=true
    echo "✓ Multi-GPU script available"
else
    USE_MULTI_GPU=false
    echo "⚠ Multi-GPU script not found, using single GPU"
fi

echo ""
echo "================================================================================"
echo "RUNNING OMEGAFOLD"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ "$USE_MULTI_GPU" = true ]; then
    # Use multi-GPU script
    echo "Using multi-GPU parallelization..."
    
    # Convert space-separated GPUs to array
    IFS=' ' read -ra GPU_ARRAY <<< "$GPUS"
    
    python bin/omegafold_across_gpus.py \
        "$INPUT_DIR"/*.fa "$INPUT_DIR"/*.fasta \
        --outdir "$OUTPUT_DIR" \
        --gpus ${GPU_ARRAY[@]}
else
    # Use single GPU with omegafold directly
    echo "Using single GPU..."
    
    # Get first GPU ID
    FIRST_GPU=$(echo $GPUS | awk '{print $1}')
    
    # Process each FASTA file
    for fasta in "$INPUT_DIR"/*.fa "$INPUT_DIR"/*.fasta; do
        if [ -f "$fasta" ]; then
            echo "Processing $(basename $fasta)..."
            CUDA_VISIBLE_DEVICES=$FIRST_GPU omegafold \
                "$fasta" \
                "$OUTPUT_DIR" \
                --device cuda:0
        fi
    done
fi

echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo ""
echo "Predicted structures saved to: $OUTPUT_DIR"
echo ""

# Count output PDB files
PDB_COUNT=$(find "$OUTPUT_DIR" -name "*.pdb" | wc -l)
echo "Generated $PDB_COUNT PDB files"
echo ""

echo "Next steps:"
echo "  1. Review predicted structures in $OUTPUT_DIR"
echo "  2. Validate structures:"
echo "     python bin/validate_structures.py \\"
echo "       --backbone_dir <original_backbones> \\"
echo "       --sequences_dir $INPUT_DIR \\"
echo "       --predicted_dir $OUTPUT_DIR \\"
echo "       --output_dir validation"
echo ""
