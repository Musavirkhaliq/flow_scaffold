# Complete Pipeline Guide - Flow Scaffolding

## Overview

This guide covers the complete workflow from training to validated protein structures:

```
Training → Sampling → ProteinMPNN → OmegaFold → Validation
```

## Quick Start - Complete Pipeline

### One Command (Automated)
```bash
bash run_complete_pipeline.sh
```

This runs all steps automatically. See below for manual step-by-step execution.

## Manual Step-by-Step

### Step 1: Train Model (2-4 hours)

```bash
bash train_and_evaluate_enhanced_flow.sh
```

**Output:** `results/enhanced_flow/enhanced_flow_*/`

### Step 2: Generate Backbones (5-10 minutes)

```bash
# Unconditional
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 \
  --n_samples 20 \
  --num_steps 50 \
  --output_dir samples/unconditional \
  --save_pdb --save_angles

# With motif scaffolding
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 \
  --n_samples 20 \
  --num_steps 50 \
  --motif_regions "30-50" \
  --guidance_scale 2.0 \
  --output_dir samples/single_motif \
  --save_pdb --save_angles
```

**Output:** `samples/*/pdb/*.pdb` (backbone structures)

### Step 3: Design Sequences with ProteinMPNN (10-20 minutes)

```bash
python bin/design_sequences_mpnn.py \
  --pdb_dir samples/unconditional/pdb \
  --output_dir sequences/unconditional \
  --num_seq_per_target 3 \
  --sampling_temp 0.1
```

**Output:** `sequences/*/*.fa` (FASTA files with designed sequences)

### Step 4: Predict Structures with OmegaFold (30-60 minutes)

First, install OmegaFold:
```bash
conda create -n omegafold python=3.9
conda activate omegafold
pip install omegafold
```

Then predict structures:
```bash
conda activate omegafold

python bin/omegafold_across_gpus.py \
  sequences/unconditional/*.fa \
  --outdir predictions/unconditional \
  --gpus 0
```

**Output:** `predictions/*/*.pdb` (full atom structures)

### Step 5: Validate Structures (5-10 minutes)

```bash
python bin/validate_structures.py \
  --backbone_dir samples/unconditional/pdb \
  --sequences_dir sequences/unconditional \
  --predicted_dir predictions/unconditional \
  --output_dir validation/unconditional
```

**Output:** `validation/*/validation_report.json` (metrics and analysis)

## Expected Results

### Backbone Quality
- CA-CA distances: 3.5-4.0 Å
- Bond angles: 100-130°
- Omega: ~180° (trans peptide bonds)

### Sequence Design (ProteinMPNN)
- Diverse sequences for each backbone
- Natural amino acid distributions
- Temperature controls diversity

### Structure Prediction (OmegaFold)
- Full atom structures from sequences
- Confidence scores (pLDDT)
- Multiple predictions per sequence

### Validation Metrics
- **TM-score**: >0.5 = good fold preservation
- **RMSD**: <2.0 Å = excellent, <5.0 Å = good
- **Success rate**: % of designs that fold correctly

## Directory Structure

```
flow_scaf/
├── results/
│   └── experiment_name/
│       ├── samples/              # Step 2: Generated backbones
│       │   ├── unconditional/
│       │   │   ├── pdb/          # Backbone PDB files
│       │   │   └── angles/       # Angle CSV files
│       │   └── single_motif/
│       ├── proteinmpnn/          # Step 3: Designed sequences
│       │   ├── unconditional/
│       │   │   └── *.fa          # FASTA files
│       │   └── single_motif/
│       ├── omegafold/            # Step 4: Predicted structures
│       │   ├── unconditional/
│       │   │   └── *.pdb         # Full atom PDB files
│       │   └── single_motif/
│       └── validation/           # Step 5: Validation results
│           ├── unconditional/
│           │   ├── validation_report.json
│           │   └── metrics.csv
│           └── single_motif/
```

## Configuration Options

### Sampling Parameters
- `--length`: Protein length (40-200 typical)
- `--n_samples`: Number of backbones to generate
- `--num_steps`: Quality vs speed (50=fast, 100=high quality)
- `--motif_regions`: Motif positions (e.g., "30-50" or "20-30,70-80")
- `--guidance_scale`: Conditioning strength (1.0-3.0)

### ProteinMPNN Parameters
- `--num_seq_per_target`: Sequences per backbone (1-10)
- `--sampling_temp`: Diversity (0.1=conservative, 0.5=diverse)

### OmegaFold Parameters
- `--gpus`: GPU IDs to use (e.g., "0" or "0 1 2 3")

## Troubleshooting

### Issue: OmegaFold not found
```bash
conda create -n omegafold python=3.9
conda activate omegafold
pip install omegafold
```

### Issue: ProteinMPNN not found
```bash
# Clone ProteinMPNN repository
git clone https://github.com/dauparas/ProteinMPNN.git
# Update path in design_sequences_mpnn.py
```

### Issue: Low TM-scores
- Increase `--num_steps` in sampling (50 → 100)
- Adjust `--guidance_scale` (try 1.5-3.0)
- Train for more epochs
- Use more diverse sequences (increase `--sampling_temp`)

### Issue: Out of memory
- Reduce `--n_samples`
- Use fewer GPUs for OmegaFold
- Process in smaller batches

## Performance Benchmarks

### Timing (Single GPU)
- Training: 2-4 hours
- Sampling (20 structures): 5-10 minutes
- ProteinMPNN (20 structures, 3 seq each): 10-20 minutes
- OmegaFold (60 sequences): 30-60 minutes
- Validation: 5-10 minutes
- **Total: ~3-5 hours**

### Quality Metrics (Expected)
- TM-score: 0.6-0.8 (good to excellent)
- RMSD: 2-5 Å (good)
- Success rate: 70-90%

## Advanced Usage

### Batch Processing
```bash
# Generate multiple experiments
for motif in "20-30" "40-60" "70-90"; do
    python bin/sample_enhanced_flow.py \
        --motif_regions "$motif" \
        --output_dir "samples/motif_${motif//-/_}"
done
```

### Custom Validation
```python
import json
from pathlib import Path

# Load validation results
with open('validation/unconditional/validation_report.json') as f:
    results = json.load(f)

# Filter high-quality designs
good_designs = [
    r for r in results['structures']
    if r['tm_score'] > 0.7 and r['rmsd'] < 3.0
]

print(f"Found {len(good_designs)} high-quality designs")
```

## Next Steps

After validation:
1. Select top designs based on TM-score and RMSD
2. Analyze secondary structure content
3. Check for clashes and geometric quality
4. Prepare for experimental validation
5. Order synthetic genes for top candidates

## References

- **FoldingDiff**: Original diffusion-based backbone generation
- **Enhanced Flow Matching**: 20x faster sampling
- **ProteinMPNN**: Sequence design for backbones
- **OmegaFold**: Fast structure prediction
- **TM-align**: Structure comparison metric
