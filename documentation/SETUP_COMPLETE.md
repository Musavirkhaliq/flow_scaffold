# Flow Scaffolding - Setup Complete ✓

## What's Included

The `flow_scaf` folder is a self-contained package for training and sampling protein backbones using enhanced flow matching with motif scaffolding.

### All Files Copied (25 modules + 5 scripts)

**Scripts:**
- ✓ `train_and_evaluate_enhanced_flow.sh` - Main pipeline
- ✓ `bin/train_enhanced_flow.py` - Training
- ✓ `bin/sample_enhanced_flow.py` - Sampling (with mean correction)
- ✓ `bin/convert_angles_to_pdb.py` - Angle to PDB conversion
- ✓ `bin/annot_secondary_structures.py` - Analysis
- ✓ `bin/compare_metrics.py` - Metrics

**Core Modules:**
- ✓ `foldingdiff/enhanced_models.py`
- ✓ `foldingdiff/flow_matching.py`
- ✓ `foldingdiff/flow_sampling.py`
- ✓ `foldingdiff/flow_models.py`
- ✓ `foldingdiff/enhanced_datasets.py`
- ✓ `foldingdiff/datasets.py` (with all fixes)
- ✓ `foldingdiff/embeddings.py`
- ✓ `foldingdiff/motif_scaffolding.py`
- ✓ `foldingdiff/conditional_diffusion.py`

**Supporting Modules:**
- ✓ `foldingdiff/modelling.py`
- ✓ `foldingdiff/losses.py`
- ✓ `foldingdiff/angles_and_coords.py`
- ✓ `foldingdiff/nerf.py`
- ✓ `foldingdiff/utils.py`
- ✓ `foldingdiff/beta_schedules.py`
- ✓ `foldingdiff/custom_metrics.py`
- ✓ `foldingdiff/sampling.py`
- ✓ `foldingdiff/conditional_sampling.py`
- ✓ `foldingdiff/tmalign.py`
- ✓ `foldingdiff/lddt.py`
- ✓ `foldingdiff/vdw_clashes.py`
- ✓ `foldingdiff/plotting.py`

**Documentation:**
- ✓ `README.md` - Comprehensive documentation
- ✓ `QUICKSTART.md` - Quick start guide
- ✓ `verify_setup.py` - Setup verification script

## All Bugs Fixed ✓

1. ✓ `ft_is_angular = [True, True, True, False, False, False]`
2. ✓ Dataset `feature_is_angular` correctly configured
3. ✓ Dataset wrapping logic uses proper `feature_is_angular`
4. ✓ Mean correction in sampling (omega +π, bond angles +natural means)

## Verification

Run the verification script:
```bash
cd flow_scaf
python verify_setup.py
```

Expected output:
```
✓ enhanced_datasets
✓ enhanced_models
✓ flow_matching
✓ flow_sampling
✓ angles_and_coords, nerf, utils
✓ All files present
✓ Data directory found
✓✓✓ SETUP VERIFICATION COMPLETE ✓✓✓
```

## Quick Start

```bash
cd flow_scaf

# Verify setup
python verify_setup.py

# Train and evaluate (2-4 hours)
bash train_and_evaluate_enhanced_flow.sh

# Or sample with existing model
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 \
  --n_samples 20 \
  --num_steps 50 \
  --motif_regions "30-50" \
  --guidance_scale 2.0 \
  --output_dir samples/test \
  --save_pdb \
  --save_angles
```

## Expected Results

After training and sampling, you should see:
- Valid protein structures with CA-CA distances ~3.8 Å
- Proper bond angles (tau ~110°, others ~115-121°)
- Omega ~180° for trans peptide bonds
- Structure span ~3.5 Å per residue

## Data Requirements

The scripts expect CATH data in `../data/cath/` (relative to flow_scaf folder).

If not present, download it:
```bash
cd ../data
bash download_cath.sh
```

## Status

✓ All files copied
✓ All dependencies included
✓ All bugs fixed
✓ Verification passed
✓ Ready to use!

---
**Created:** December 18, 2024
**Status:** Complete and verified
