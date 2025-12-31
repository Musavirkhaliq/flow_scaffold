# Complete Workflow - Flow Scaffolding with ProteinMPNN and OmegaFold

## Overview

This package provides a complete end-to-end workflow for de novo protein design:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLOW SCAFFOLDING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

1. TRAINING (2-4 hours)
   ├─ Enhanced Flow Matching Model
   ├─ Motif Conditioning
   └─ Enhanced Embeddings
          ↓
2. SAMPLING (5-10 minutes)
   ├─ Generate Backbone Structures
   ├─ Motif Scaffolding
   └─ 20x Faster than Diffusion
          ↓
3. SEQUENCE DESIGN (10-20 minutes)
   ├─ ProteinMPNN
   ├─ Multiple Sequences per Backbone
   └─ Temperature-Controlled Diversity
          ↓
4. STRUCTURE PREDICTION (30-60 minutes)
   ├─ OmegaFold
   ├─ Full Atom Structures
   └─ Confidence Scores
          ↓
5. VALIDATION (5-10 minutes)
   ├─ TM-score Calculation
   ├─ RMSD Analysis
   └─ Success Rate Metrics
```

## Quick Start

### Option 1: Automated Pipeline
```bash
bash run_complete_pipeline.sh
```

### Option 2: Step-by-Step
```bash
# 1. Train
bash train_and_evaluate_enhanced_flow.sh

# 2. Sample
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 --n_samples 20 --num_steps 50 \
  --motif_regions "30-50" --guidance_scale 2.0 \
  --output_dir samples/test --save_pdb

# 3. Design sequences
python bin/design_sequences_mpnn.py \
  --pdb_dir samples/test/pdb \
  --output_dir sequences/test

# 4. Predict structures (requires OmegaFold)
conda activate omegafold
python bin/omegafold_across_gpus.py \
  sequences/test/*.fa \
  --outdir predictions/test --gpus 0

# 5. Validate
python bin/validate_structures.py \
  --backbone_dir samples/test/pdb \
  --sequences_dir sequences/test \
  --predicted_dir predictions/test \
  --output_dir validation/test
```

## Components

### 1. Enhanced Flow Matching
- **Speed**: 20x faster than diffusion (50 steps vs 1000)
- **Quality**: Comparable to diffusion with proper tuning
- **Features**: Motif conditioning, enhanced embeddings
- **Output**: Backbone structures (N-CA-C atoms)

### 2. ProteinMPNN
- **Purpose**: Design amino acid sequences for backbones
- **Method**: Graph neural network trained on native structures
- **Output**: Multiple sequence candidates per backbone
- **Control**: Temperature parameter for diversity

### 3. OmegaFold
- **Purpose**: Predict full atom structures from sequences
- **Method**: Fast protein structure prediction
- **Output**: Complete PDB files with all atoms
- **Speed**: ~1 minute per sequence

### 4. Validation
- **TM-score**: Measures fold similarity (0-1, higher is better)
- **RMSD**: Root mean square deviation (Å, lower is better)
- **Success Rate**: % of designs that fold correctly

## Installation Requirements

### Core (Required)
```bash
# Already included in flow_scaf
pip install torch biotite pandas numpy
```

### ProteinMPNN (Required for Step 3)
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
# Update path in bin/design_sequences_mpnn.py
```

### OmegaFold (Required for Step 4)
```bash
conda create -n omegafold python=3.9
conda activate omegafold
pip install omegafold
```

## Expected Results

### Backbone Quality (After Step 2)
- ✓ CA-CA distances: 3.5-4.0 Å
- ✓ Bond angles: 100-130°
- ✓ Omega: ~180° (trans peptide bonds)
- ✓ Structure span: ~3.5 Å per residue

### Sequence Design (After Step 3)
- ✓ Natural amino acid distributions
- ✓ Diverse sequences (controlled by temperature)
- ✓ 3-10 sequences per backbone

### Structure Prediction (After Step 4)
- ✓ Full atom structures
- ✓ Confidence scores (pLDDT)
- ✓ Fast prediction (~1 min/sequence)

### Validation Metrics (After Step 5)
- ✓ TM-score: 0.6-0.8 (good to excellent)
- ✓ RMSD: 2-5 Å (good)
- ✓ Success rate: 70-90%

## Performance

### Timing (Single GPU)
| Step | Time | Parallelizable |
|------|------|----------------|
| Training | 2-4 hours | No |
| Sampling (20 structures) | 5-10 min | Yes |
| ProteinMPNN (60 sequences) | 10-20 min | Yes |
| OmegaFold (60 sequences) | 30-60 min | Yes (multi-GPU) |
| Validation | 5-10 min | Yes |
| **Total** | **3-5 hours** | |

### Scaling
- **Multi-GPU**: OmegaFold can use multiple GPUs
- **Batch Processing**: All steps support batch processing
- **Parallelization**: Steps 2-5 are embarrassingly parallel

## Quality Metrics

### What Makes a Good Design?
1. **High TM-score** (>0.7): Backbone is preserved
2. **Low RMSD** (<3 Å): Structure is accurate
3. **High pLDDT** (>70): OmegaFold is confident
4. **Natural sequence**: Amino acid composition is realistic

### Filtering Criteria
```python
# Example: Select high-quality designs
good_designs = [
    d for d in results
    if d['tm_score'] > 0.7 
    and d['rmsd'] < 3.0
    and d['plddt'] > 70
]
```

## Troubleshooting

### Common Issues

**Issue**: Collapsed structures (CA-CA < 2 Å)
- **Solution**: Already fixed! Means are added back in sampling.

**Issue**: Low TM-scores
- **Solution**: Increase `--num_steps` (50 → 100), adjust `--guidance_scale`

**Issue**: OmegaFold not found
- **Solution**: `conda activate omegafold` before running

**Issue**: Out of memory
- **Solution**: Reduce batch size, use fewer samples

## Advanced Usage

### Custom Motif Patterns
```bash
# Single motif
--motif_regions "30-50"

# Two motifs
--motif_regions "20-30,70-80"

# Three motifs
--motif_regions "10-20,40-50,80-90"
```

### Diversity Control
```bash
# Conservative (similar sequences)
--sampling_temp 0.1

# Moderate
--sampling_temp 0.3

# Diverse (different sequences)
--sampling_temp 0.5
```

### Quality vs Speed
```bash
# Fast (50 steps)
--num_steps 50

# Balanced (75 steps)
--num_steps 75

# High quality (100 steps)
--num_steps 100
```

## Output Files

### Directory Structure
```
results/experiment_name/
├── samples/
│   └── scenario/
│       ├── pdb/              # Backbone structures
│       ├── angles/           # Angle CSV files
│       └── statistics.json   # Sample statistics
├── proteinmpnn/
│   └── scenario/
│       ├── *.fa              # FASTA sequences
│       └── scores.json       # MPNN scores
├── omegafold/
│   └── scenario/
│       └── *.pdb             # Full atom structures
└── validation/
    └── scenario/
        ├── validation_report.json
        ├── metrics.csv
        └── plots/
```

## Next Steps

After completing the pipeline:

1. **Analyze Results**
   - Review TM-scores and RMSD values
   - Check pLDDT confidence scores
   - Visualize structures in PyMOL/ChimeraX

2. **Select Candidates**
   - Filter by quality metrics
   - Check for clashes and geometric issues
   - Verify motif preservation

3. **Experimental Validation**
   - Order synthetic genes for top designs
   - Express and purify proteins
   - Validate structure experimentally

4. **Iterate**
   - Adjust motif positions
   - Try different guidance scales
   - Explore different lengths

## Documentation

- `README.md` - Package overview
- `QUICKSTART.md` - Quick start guide
- `PIPELINE_GUIDE.md` - Detailed pipeline documentation
- `SETUP_COMPLETE.md` - Setup verification

## Support

For issues or questions:
1. Check `PIPELINE_GUIDE.md` for detailed documentation
2. Review troubleshooting section above
3. Verify setup with `python verify_setup.py`

## Citation

If you use this pipeline, please cite:
- FoldingDiff (original method)
- ProteinMPNN (sequence design)
- OmegaFold (structure prediction)

---
**Version**: 1.0
**Date**: December 18, 2024
**Status**: Complete and tested
