#!/bin/bash
# Quick setup script for reference structures
# Usage: ./evaluations/setup_references.sh [number_of_references] [source_dir]

set -e

# Default values
N_REFERENCES=${1:-25}
CATH_DIR="${2:-data/cath/dompdb}"
REF_DIR="data/references"

echo "================================================================================"
echo "Setting up Reference Structures"
echo "================================================================================"
echo ""
echo "Number of references: $N_REFERENCES"
echo "Source directory: $CATH_DIR"
echo "Target directory: $REF_DIR"
echo ""

# Check if CATH directory exists
if [ ! -d "$CATH_DIR" ]; then
    echo "❌ Error: CATH directory not found: $CATH_DIR"
    echo ""
    echo "Available options:"
    echo "  1. Use CATH database: $CATH_DIR"
    echo "  2. Use custom directory: ./evaluations/setup_references.sh $N_REFERENCES /path/to/structures"
    exit 1
fi

# Count available structures
N_AVAILABLE=$(ls "$CATH_DIR" | wc -l)
echo "Available structures in CATH: $N_AVAILABLE"

if [ "$N_REFERENCES" -gt "$N_AVAILABLE" ]; then
    echo "⚠️  Warning: Requested $N_REFERENCES references but only $N_AVAILABLE available"
    N_REFERENCES=$N_AVAILABLE
fi

# Create reference directory
mkdir -p "$REF_DIR"
echo "Created directory: $REF_DIR"

# Copy structures
echo ""
echo "Copying structures..."
COPIED=0
ls "$CATH_DIR" | head -$N_REFERENCES | while read f; do
    src="$CATH_DIR/$f"
    dst="$REF_DIR/${f}.pdb"
    
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        COPIED=$((COPIED + 1))
        if [ $((COPIED % 5)) -eq 0 ]; then
            echo "  Copied $COPIED/$N_REFERENCES structures..."
        fi
    fi
done

# Count copied files
N_COPIED=$(ls "$REF_DIR" | wc -l)
echo ""
echo "✅ Copied $N_COPIED structures to $REF_DIR"

# Option to rename to match generated structures
echo ""
read -p "Rename references to match generated structures (sample_XXXX.pdb)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Renaming structures..."
    cd "$REF_DIR"
    i=0
    for f in $(ls | head -$N_REFERENCES); do
        new_name="sample_$(printf "%04d" $i).pdb"
        mv "$f" "$new_name"
        i=$((i + 1))
    done
    echo "✅ Renamed $i structures"
fi

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "Reference structures are ready at: $REF_DIR"
echo ""
echo "You can now run evaluation with:"
echo ""
echo "  python evaluations/evaluate_sampled_backbones.py \\"
echo "      --samples_dir results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \\"
echo "      --reference_dir $REF_DIR"
echo ""
echo "Or use the quick evaluation script:"
echo ""
echo "  python evaluations/run_evaluation.py \\"
echo "      results/advanced_flow/samples_advanced_flow_260102_022234/two_motifs_short \\"
echo "      --reference_dir $REF_DIR"
echo ""

