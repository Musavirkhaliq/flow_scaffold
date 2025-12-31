# Fix: Repetitive ProteinMPNN Sequences

## Problem Identified

ProteinMPNN is generating repetitive sequences (e.g., long stretches of G, H, A) because **the generated backbones are too uniform**.

### Root Cause

The backbone structures have:
- ✓ Correct CA-CA distances (~3.8 Å)
- ✗ **Too little variation** (std = 0.02-0.04 Å, should be 0.2-0.4 Å)
- ✗ **Too regular geometry** (all CA-CA distances nearly identical)

This uniformity makes ProteinMPNN think the backbone is unnatural, so it assigns repetitive/low-confidence sequences.

## Diagnosis

Run the diagnostic tool:
```bash
python diagnose_backbones.py results/*/samples/*/pdb/*.pdb
```

Look for:
- CA-CA std < 0.1 Å = CRITICAL (will cause repetitive sequences)
- CA-CA std < 0.2 Å = WARNING (may cause issues)

## Solutions

### Solution 1: Improve Sampling (Recommended)

Generate more diverse backbones by adjusting sampling parameters:

```bash
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 \
  --n_samples 20 \
  --num_steps 100 \
  --guidance_scale 1.5 \
  --output_dir samples/diverse \
  --save_pdb
```

**Key changes:**
- `--num_steps 100` (was 50) - More steps = more diversity
- `--guidance_scale 1.5` (was 2.0) - Lower guidance = more diversity

### Solution 2: Increase ProteinMPNN Temperature (Quick Fix)

Use higher temperature to force more diverse sequences:

```bash
bash run_proteinmpnn.sh \
  --input_dir samples/test/pdb \
  --output_dir sequences/diverse \
  --num_sequences 10 \
  --temperature 0.5
```

**Temperature guide:**
- 0.1 = Conservative (current, causes repetition with uniform backbones)
- 0.3 = Moderate diversity
- 0.5 = High diversity (recommended for uniform backbones)
- 1.0 = Maximum diversity

**Trade-off**: Higher temperature = more diverse but lower confidence scores

### Solution 3: Train Longer (Long-term)

The model needs more training to learn diverse backbone geometries:

```bash
# Edit train_and_evaluate_enhanced_flow.sh
# Change: --epochs 2
# To: --epochs 10-20

bash train_and_evaluate_enhanced_flow.sh
```

This will take longer but produce better quality backbones.

## Quick Test

Test if higher temperature helps:

```bash
# 1. Take one existing backbone
cp samples/test/pdb/sample_0000.pdb /tmp/test.pdb

# 2. Try different temperatures
for temp in 0.1 0.3 0.5; do
    echo "Testing temperature $temp..."
    python bin/design_sequences_mpnn.py \
        --input_dir /tmp \
        --output_dir /tmp/test_temp_$temp \
        --num_sequences 3 \
        --temperature $temp \
        --max_structures 1
    
    # Check first sequence
    head -2 /tmp/test_temp_$temp/test/seqs/test.fa
    echo ""
done
```

Compare the sequences - higher temperature should show more diversity.

## Expected Results

### Before Fix (T=0.1, uniform backbone)
```
>sample
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
>sample
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
```

### After Fix (T=0.5 or better sampling)
```
>sample
MKLVDAEWRQHGPNSTYFICVLAEQWRGHPNSTYFICVLAEWRQHGP
>sample
AKLVDGEWRQHGPNSTYFICVLAEQWRGHPNSTYFICVLAEWRQHGP
```

## Verification

After applying fixes, check backbone quality:

```bash
python diagnose_backbones.py samples/diverse/pdb/*.pdb
```

Look for:
- CA-CA std > 0.2 Å ✓
- Angle std > 15° ✓
- No "Too uniform" warnings ✓

Then check ProteinMPNN sequences:

```bash
# Should see diverse amino acids, not repetitive
head sequences/diverse/*/seqs/*.fa
```

## Why This Happens

The flow matching model learns to generate smooth, regular backbones because:
1. Training data has some regularity
2. Flow matching naturally produces smooth trajectories
3. Not enough training epochs to learn full diversity
4. Guidance can over-regularize structures

This is a known limitation of generative models - they need careful tuning to balance:
- **Validity** (correct geometry) ✓ We have this
- **Diversity** (varied structures) ✗ Need to improve
- **Designability** (ProteinMPNN likes them) ✗ Consequence of low diversity

## Recommended Workflow

1. **Diagnose** current backbones:
   ```bash
   python diagnose_backbones.py samples/*/pdb/*.pdb
   ```

2. **If too uniform**, resample with better parameters:
   ```bash
   python bin/sample_enhanced_flow.py \
     --num_steps 100 --guidance_scale 1.5 ...
   ```

3. **Use higher ProteinMPNN temperature**:
   ```bash
   bash run_proteinmpnn.sh --temperature 0.5 ...
   ```

4. **Validate** sequences are diverse:
   ```bash
   head sequences/*/seqs/*.fa
   ```

5. **If still issues**, train longer:
   ```bash
   # Increase epochs in training script
   bash train_and_evaluate_enhanced_flow.sh
   ```

## Summary

- ✓ Backbones have correct CA-CA distances
- ✗ Backbones are too uniform (low std)
- → ProteinMPNN generates repetitive sequences
- **Fix**: Increase sampling diversity OR ProteinMPNN temperature
- **Long-term**: Train model longer for better diversity

---
**Status**: Issue identified and solutions provided  
**Quick fix**: Use `--temperature 0.5` with ProteinMPNN  
**Better fix**: Resample with `--num_steps 100 --guidance_scale 1.5`
