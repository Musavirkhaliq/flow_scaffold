# Setup Guide - Flow Scaffolding

## System Configuration

This guide is configured for your specific setup:

- **ProteinMPNN**: `/home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN`
- **OmegaFold**: Conda environment `omegafold`

## Quick Setup Verification

```bash
cd flow_scaf

# 1. Verify Python packages
python verify_setup.py

# 2. Check ProteinMPNN
ls /home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN/protein_mpnn_run.py

# 3. Check OmegaFold
conda activate omegafold
which omegafold
```

## Dependencies Status

### ✓ Already Installed

**ProteinMPNN**
- Location: `/home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN`
- Status: ✓ Installed
- No action needed

**OmegaFold**
- Environment: `omegafold`
- Status: ✓ Installed
- Activation: `conda activate omegafold`

### Core Python Packages

Check if installed:
```bash
python -c "import torch; print('✓ PyTorch')"
python -c "import biotite; print('✓ biotite')"
python -c "import pandas; print('✓ pandas')"
python -c "import numpy; print('✓ numpy')"
```

If missing, install:
```bash
pip install torch biotite pandas numpy
```

## Usage with Your Setup

### Option 1: Complete Pipeline

**Important**: Activate omegafold environment first!

```bash
conda activate omegafold
cd flow_scaf
bash run_complete_pipeline.sh
```

### Option 2: Step-by-Step

```bash
cd flow_scaf

# 1. Training (no special environment needed)
bash train_and_evaluate_enhanced_flow.sh

# 2. Sampling (no special environment needed)
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 --n_samples 20 --save_pdb

# 3. ProteinMPNN (no special environment needed)
bash run_proteinmpnn.sh --input_dir samples/*/pdb

# 4. OmegaFold (REQUIRES omegafold environment)
conda activate omegafold
bash run_omegafold.sh --input_dir sequences
```

### Option 3: Individual Tools

**ProteinMPNN only:**
```bash
cd flow_scaf
bash run_proteinmpnn.sh \
  --input_dir <your_pdb_dir> \
  --output_dir sequences \
  --num_sequences 5
```

**OmegaFold only:**
```bash
conda activate omegafold
cd flow_scaf
bash run_omegafold.sh \
  --input_dir <your_fasta_dir> \
  --output_dir predictions \
  --gpus "0"
```

## Environment Management

### Switching Environments

```bash
# For training and sampling (base environment)
conda deactivate  # or use your default environment

# For OmegaFold
conda activate omegafold
```

### Checking Current Environment

```bash
echo $CONDA_DEFAULT_ENV
```

Should show:
- `omegafold` when running OmegaFold
- Your base environment for other steps

## Troubleshooting

### ProteinMPNN Issues

**Error: ProteinMPNN not found**
```bash
# Check if it exists
ls /home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN/protein_mpnn_run.py

# If missing, it's in the wrong location
# Update the path in flow_scaf/bin/design_sequences_mpnn.py
```

**Error: No module named 'torch'**
```bash
# Install PyTorch
pip install torch
```

### OmegaFold Issues

**Error: OmegaFold environment not activated**
```bash
# Activate the environment
conda activate omegafold

# Verify
echo $CONDA_DEFAULT_ENV  # Should show: omegafold
```

**Error: omegafold command not found**
```bash
# Make sure you're in the right environment
conda activate omegafold

# Install OmegaFold
pip install omegafold

# Verify
which omegafold
```

### General Issues

**Import errors**
```bash
# Run verification
python verify_setup.py

# Check specific import
python -c "from foldingdiff.enhanced_models import BertForFlowMatchingEnhanced"
```

**Path issues**
```bash
# Make sure you're in the flow_scaf directory
cd flow_scaf
pwd  # Should end with: /flow_scaf
```

## Quick Reference

### Environment Cheat Sheet

| Task | Environment | Command |
|------|-------------|---------|
| Training | Base | `bash train_and_evaluate_enhanced_flow.sh` |
| Sampling | Base | `python bin/sample_enhanced_flow.py ...` |
| ProteinMPNN | Base | `bash run_proteinmpnn.sh ...` |
| OmegaFold | omegafold | `conda activate omegafold && bash run_omegafold.sh ...` |
| Validation | Base | `python bin/validate_structures.py ...` |

### Path Cheat Sheet

| Component | Path |
|-----------|------|
| ProteinMPNN | `/home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN` |
| Flow Scaf | `/home/musa/Documents/augment-projects/foldingdiff/flow_scaf` |
| Data | `/home/musa/Documents/augment-projects/foldingdiff/data/cath` |

## Verification Checklist

Run these commands to verify everything is set up:

```bash
cd flow_scaf

# 1. Python packages
python verify_setup.py

# 2. ProteinMPNN
ls /home/musa/Documents/augment-projects/foldingdiff/software/ProteinMPNN/protein_mpnn_run.py && echo "✓ ProteinMPNN found"

# 3. OmegaFold
conda activate omegafold
which omegafold && echo "✓ OmegaFold found"
conda deactivate

# 4. Data
ls ../data/cath/dompdb/*.pdb | head -5 && echo "✓ Data found"
```

Expected output:
```
✓ All imports successful
✓ ProteinMPNN found
✓ OmegaFold found
✓ Data found
```

## Ready to Use!

Once all checks pass, you're ready to run:

```bash
# Complete pipeline (with OmegaFold)
conda activate omegafold
bash run_complete_pipeline.sh

# Or without OmegaFold
bash run_complete_pipeline.sh
# (will skip OmegaFold automatically)
```

---
**Date**: December 18, 2024  
**Status**: Configured for your system
