#!/bin/bash
# Standalone script to run ProteinMPNN sequence design on generated backbones

set -e

echo "================================================================================"
echo "PROTEINMPNN SEQUENCE DESIGN - STANDALONE"
echo "================================================================================"
echo ""

# Default parameters
INPUT_DIR=""
OUTPUT_DIR="sequences"
NUM_SEQUENCES=3
TEMPERATURE=0.3  # Increased from 0.1 to reduce repetitive sequences
MAX_STRUCTURES=""

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
        --num_sequences)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_structures)
            MAX_STRUCTURES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_proteinmpnn.sh --input_dir <pdb_dir> [options]"
            echo ""
            echo "Required:"
            echo "  --input_dir DIR          Directory containing PDB files"
            echo ""
            echo "Optional:"
            echo "  --output_dir DIR         Output directory (default: sequences)"
            echo "  --num_sequences N        Sequences per structure (default: 3)"
            echo "  --temperature T          Sampling temperature 0.1-1.0 (default: 0.3)"
            echo "                           0.1 = very conservative (may cause repetition)"
            echo "                           0.3 = balanced (recommended)"
            echo "                           0.5-1.0 = diverse"
            echo "  --max_structures N       Max structures to process (default: all)"
            echo ""
            echo "Example:"
            echo "  bash run_proteinmpnn.sh --input_dir samples/test/pdb --num_sequences 5 --temperature 0.3"
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

# Check for PDB files
PDB_COUNT=$(find "$INPUT_DIR" -name "*.pdb" | wc -l)
if [ "$PDB_COUNT" -eq 0 ]; then
    echo "Error: No PDB files found in $INPUT_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Input directory:  $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  PDB files found:  $PDB_COUNT"
echo "  Sequences/struct: $NUM_SEQUENCES"
echo "  Temperature:      $TEMPERATURE"
if [ -n "$MAX_STRUCTURES" ]; then
    echo "  Max structures:   $MAX_STRUCTURES"
fi
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "✗ Python not found"
    exit 1
fi
echo "✓ Python found"

# Check PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo "✗ PyTorch not found"
    echo ""
    echo "ProteinMPNN requires PyTorch. Install with:"
    echo "  pip install torch"
    exit 1
fi
echo "✓ PyTorch found"

# Check ProteinMPNN
MPNN_FOUND=false
for mpnn_path in "/home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN" "../software/ProteinMPNN" "../../software/ProteinMPNN" "software/ProteinMPNN" "ProteinMPNN"; do
    if [ -f "$mpnn_path/protein_mpnn_run.py" ]; then
        echo "✓ ProteinMPNN found at: $mpnn_path"
        MPNN_FOUND=true
        break
    fi
done

if [ "$MPNN_FOUND" = false ]; then
    echo "✗ ProteinMPNN not found"
    echo ""
    echo "Please install ProteinMPNN:"
    echo "  git clone https://github.com/dauparas/ProteinMPNN.git"
    exit 1
fi

echo ""
echo "================================================================================"
echo "RUNNING PROTEINMPNN"
echo "================================================================================"
echo ""

# Build command
CMD="python bin/design_sequences_mpnn.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_sequences $NUM_SEQUENCES \
    --temperature $TEMPERATURE"

if [ -n "$MAX_STRUCTURES" ]; then
    CMD="$CMD --max_structures $MAX_STRUCTURES"
fi

# Run ProteinMPNN
eval $CMD

echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo ""
echo "Sequences saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review designed sequences in $OUTPUT_DIR"
echo "  2. Predict structures with OmegaFold:"
echo "     bash run_omegafold.sh --input_dir $OUTPUT_DIR"
echo ""
