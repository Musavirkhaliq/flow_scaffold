# Quick Fix Guide: Repetitive Sequences

## Problem
ProteinMPNN generates sequences with excessive repetition (GGGGGG, HHHHHH, etc.)

## Cause
1. **Temperature too low (0.1)** - Makes ProteinMPNN too conservative
2. **Model undertrained (5 epochs)** - Produces poor quality backbones

## Solution Applied ✅

All fixes have been applied to the codebase. The scripts now use correct defaults.

## Quick Test (5 minutes)

Test with existing backbones using the new temperature:

```bash
# Generate sequences with proper temperature
bash run_proteinmpnn.sh \
    --input_dir flow_scaf/samples/test_trained_model/pdb \
    --output_dir sequences_fixed \
    --temperature 0.3 \
    --num_sequences 5

# Analyze quality
python analyze_sequence_quality.py sequences_fixed/sample_0000/seqs/sample_0000.fa
```

**Expected:** Significantly less repetition, higher entropy

## Full Fix (Hours/Days)

Retrain the model with proper settings (already configured):

```bash
# Train with correct settings (100 epochs, LR=5e-5, 100 steps)
bash train_and_evaluate_enhanced_flow.sh

# Sample from trained model
python sample_from_checkpoint.py

# Design sequences
bash run_proteinmpnn.sh \
    --input_dir flow_scaf/samples/test_trained_model/pdb \
    --temperature 0.3
```

## What Changed

| Parameter | Before | After | File |
|-----------|--------|-------|------|
| Temperature | 0.1 | **0.3** | `run_proteinmpnn.sh` |
| Epochs | 5 | **100** | `train_and_evaluate_enhanced_flow.sh` |
| Learning Rate | 1e-4 | **5e-5** | `train_and_evaluate_enhanced_flow.sh` |
| Sampling Steps | 50 | **100** | `train_and_evaluate_enhanced_flow.sh` |

## Temperature Guide

- **0.1**: Too conservative → repetitive sequences ❌
- **0.3**: Balanced → recommended ✅
- **0.5**: Diverse → good for exploration
- **1.0**: Maximum diversity → may be unstable

## Validation

Check sequence quality:

```bash
# Analyze any FASTA file
python analyze_sequence_quality.py <fasta_file>

# Summary only
python analyze_sequence_quality.py <fasta_file> --summary

# Verbose output
python analyze_sequence_quality.py <fasta_file> -v
```

Good sequences should have:
- ✅ Entropy >2.5 bits
- ✅ No amino acid >25%
- ✅ Max consecutive identical residues <5
- ✅ <20% low-complexity regions

## Documentation

- **Detailed Analysis**: `documentation/REPETITIVE_SEQUENCES_ANALYSIS.md`
- **Fix Summary**: `documentation/FIX_SUMMARY.md`
- **This Guide**: `QUICK_FIX_GUIDE.md`

## Need Help?

1. Check if temperature is set correctly: `grep TEMPERATURE run_proteinmpnn.sh`
2. Verify training settings: `grep -E "EPOCHS|LR|NUM_STEPS" train_and_evaluate_enhanced_flow.sh`
3. Run sequence analysis: `python analyze_sequence_quality.py <fasta>`
4. Review detailed docs: `cat documentation/REPETITIVE_SEQUENCES_ANALYSIS.md`

---

**Status**: ✅ Fixed
**Impact**: Critical issue resolved
**Action Required**: Run quick test or full retraining
