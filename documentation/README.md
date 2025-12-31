# Flow Scaffolding - Enhanced Flow Matching for Protein Design

This folder contains all the necessary files for training and sampling protein backbones using enhanced flow matching with motif scaffolding.

## Contents

### Core Scripts
- `train_and_evaluate_enhanced_flow.sh` - Main training and evaluation pipeline
- `bin/train_enhanced_flow.py` - Training script
- `bin/sample_enhanced_flow.py` - Sampling script with mean correction
- `bin/convert_angles_to_pdb.py` - Convert angles to PDB structures

### Core Modules
- `foldingdiff/enhanced_models.py` - Enhanced flow matching model
- `foldingdiff/flow_matching.py` - Flow matching implementation
- `foldingdiff/flow_sampling.py` - Flow matching sampling
- `foldingdiff/enhanced_datasets.py` - Enhanced dataset with embeddings
- `foldingdiff/datasets.py` - Base dataset classes (with fixes)
- `foldingdiff/embeddings.py` - Enhanced embeddings (coords, frames, pairwise)
- `foldingdiff/motif_scaffolding.py` - Motif conditioning utilities

### Supporting Modules
- `foldingdiff/modelling.py` - Base model classes
- `foldingdiff/losses.py` - Loss functions
- `foldingdiff/angles_and_coords.py` - Angle/coordinate conversions
- `foldingdiff/nerf.py` - NERF coordinate generation
- `foldingdiff/utils.py` - Utility functions
- `foldingdiff/beta_schedules.py` - Beta schedules for diffusion
- `foldingdiff/custom_metrics.py` - Custom metrics (wrapped mean, etc.)
- `foldingdiff/sampling.py` - Sampling utilities
- `foldingdiff/conditional_sampling.py` - Conditional sampling
- `foldingdiff/tmalign.py` - TM-align utilities
- `foldingdiff/lddt.py` - lDDT metric
- `foldingdiff/vdw_clashes.py` - VDW clash detection
- `foldingdiff/plotting.py` - Plotting utilities

### Analysis Tools
- `bin/annot_secondary_structures.py` - Secondary structure annotation
- `bin/compare_metrics.py` - Metrics comparison

### Pipeline Tools
- `bin/design_sequences_mpnn.py` - ProteinMPNN sequence design
- `bin/omegafold_across_gpus.py` - OmegaFold structure prediction
- `bin/validate_structures.py` - Structure validation
- `run_complete_pipeline.sh` - Complete automated pipeline

## Key Features

### All Bugs Fixed ✓
1. ✓ `ft_is_angular` correctly set to `[True, True, True, False, False, False]`
2. ✓ Dataset `feature_is_angular` correctly configured
3. ✓ Dataset wrapping logic uses proper `feature_is_angular` (not colon counting)
4. ✓ Mean correction applied after sampling (omega +π, bond angles +natural means)

### Enhanced Flow Matching
- 20x faster than diffusion (50 steps vs 1000)
- Conditional motif scaffolding
- Enhanced embeddings (coordinates, local frames, pairwise distances)
- Classifier-free guidance

## Usage

### Complete Pipeline (Recommended)
```bash
bash run_complete_pipeline.sh
```

This runs the complete workflow:
1. Train enhanced flow matching model (2-4 hours)
2. Generate backbone structures (5-10 min)
3. Design sequences with ProteinMPNN (10-20 min)
4. Predict structures with OmegaFold (30-60 min)
5. Validate and analyze results (5-10 min)

See `PIPELINE_GUIDE.md` for detailed documentation.

### Training Only
```bash
bash train_and_evaluate_enhanced_flow.sh
```

This will:
1. Train enhanced flow matching model (2-4 hours)
2. Generate samples with motif conditioning
3. Evaluate structure quality
4. Compare with baseline metrics

### Sampling Only
```bash
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 \
  --n_samples 20 \
  --num_steps 50 \
  --motif_regions "30-50" \
  --guidance_scale 2.0 \
  --output_dir samples/my_samples \
  --save_pdb \
  --save_angles
```

### Convert Existing Angles to PDB
```bash
python bin/convert_angles_to_pdb.py \
  --input_dir path/to/angles \
  --output_dir path/to/pdb \
  --pattern "*.csv"
```

## Important Notes

### Mean Correction
The model learns mean-centered angles, so we add means back during PDB generation:
- Omega: +π (180°) for trans peptide bonds
- Tau: +1.92 rad (~110°)
- CA:C:1N: +2.01 rad (~115°)
- C:1N:1CA: +2.11 rad (~121°)

This is already implemented in `sample_enhanced_flow.py` and `convert_angles_to_pdb.py`.

### Expected Results
- CA-CA distances: ~3.8 Å
- Structure span: ~3.5 Å per residue
- Bond angles: 100-130°
- Omega: ~180° (trans peptide bonds)

### Data Requirements
Requires CATH dataset in `../data/cath/` (relative to this folder).

## File Structure
```
flow_scaf/
├── README.md (this file)
├── QUICKSTART.md
├── PIPELINE_GUIDE.md
├── train_and_evaluate_enhanced_flow.sh
├── run_complete_pipeline.sh
├── bin/
│   ├── train_enhanced_flow.py
│   ├── sample_enhanced_flow.py
│   ├── convert_angles_to_pdb.py
│   ├── design_sequences_mpnn.py
│   ├── omegafold_across_gpus.py
│   ├── validate_structures.py
│   ├── annot_secondary_structures.py
│   └── compare_metrics.py
└── foldingdiff/
    ├── __init__.py
    ├── enhanced_models.py
    ├── flow_matching.py
    ├── flow_sampling.py
    ├── enhanced_datasets.py
    ├── datasets.py
    ├── embeddings.py
    ├── motif_scaffolding.py
    ├── modelling.py
    ├── losses.py
    ├── angles_and_coords.py
    ├── nerf.py
    ├── utils.py
    ├── beta_schedules.py
    ├── custom_metrics.py
    ├── sampling.py
    ├── conditional_sampling.py
    ├── tmalign.py
    ├── lddt.py
    ├── vdw_clashes.py
    └── plotting.py
```

## Version
All files include the complete bug fixes as of December 18, 2024.

## References
- Enhanced Flow Matching: 20x faster than diffusion
- Motif Scaffolding: Conditional generation with guidance
- Enhanced Embeddings: Geometric features for better quality
