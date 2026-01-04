#!/bin/bash
# Download AlphaFold subset for training
# Recommended: Download human + model organisms for ~30K structures

set -e

ALPHAFOLD_DIR="${1:-data/alphafold}"
BASE_URL="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"

echo "=========================================="
echo "Downloading AlphaFold Subset"
echo "=========================================="
echo "Target directory: $ALPHAFOLD_DIR"
echo ""

mkdir -p "$ALPHAFOLD_DIR"
cd "$ALPHAFOLD_DIR"

# Organisms to download (can be customized)
ORGANISMS=(
    "UP000005640_9606_HUMAN"      # Human (~20K structures)
    "UP000000625_83333_ECOLI"     # E. coli (~4K structures)
    "UP000002311_559292_YEAST"    # Yeast (~6K structures)
    "UP000000589_10090_MOUSE"     # Mouse (~20K structures)
)

echo "Downloading from ${#ORGANISMS[@]} organisms..."
echo ""

for org in "${ORGANISMS[@]}"; do
    echo "Downloading $org..."
    wget -r -np -nH --cut-dirs=2 \
        "${BASE_URL}/${org}/" \
        --accept "*.pdb.gz" \
        --no-directories \
        --continue \
        --progress=bar:force 2>&1 | grep -E "(saved|already|error)" || true
    
    echo "✓ Completed $org"
    echo ""
done

# Count downloaded files
TOTAL_FILES=$(find . -name "*.pdb.gz" | wc -l)
TOTAL_SIZE=$(du -sh . | cut -f1)

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Total files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"
echo ""
echo "Next steps:"
echo "1. Verify files: ls $ALPHAFOLD_DIR | head -10"
echo "2. Update training script to use combined dataset"
echo "3. Re-train model with larger dataset"



