# Fix Plan: Critical Issues

**Date:** 2026-01-02  
**Issues:** Clash rate doubled, Omega fix regressed, Quality improvement too small

---

## Critical Issues Identified

1. **Clash Rate Doubled** (0.391 → 0.804)
2. **Omega Fix Regressed** (99.4% trans → 82.7% trans)
3. **Quality Improvement Too Small** (only 4.4% vs expected 3-4x)

---

## Fix 1: Ensure Omega Fix is Applied Correctly

### Problem
Omega fix is applied at line 741, but may be getting undone or not applied in all code paths.

### Solution
1. **Verify omega fix is applied after mean correction** (already done at line 741)
2. **Add explicit check** to ensure omega is π before structure conversion
3. **Add logging** to verify omega values

### Code Changes
```python
# In bin/sample_advanced_flow.py, after line 741
# CRITICAL: Fix omega angles to trans (π) AFTER applying means
sample_corrected_np[:, 2] = np.pi

# ADD: Verify omega is correct
omega_values = sample_corrected_np[:, 2]
if not np.allclose(omega_values, np.pi, atol=0.01):
    logger.warning(f"⚠️  Omega not correctly set! Values: {omega_values[:5]}")
    # Force fix
    sample_corrected_np[:, 2] = np.pi
```

---

## Fix 2: Reduce Geometric Loss Weight

### Problem
Geometric loss weight (0.30) may be too high, causing training instability and high clash rates.

### Solution
Reduce geometric loss weight from 0.30 to 0.20-0.25 for more stable training.

### Code Changes
```python
# In bin/train_advanced_flow.py, line 66
parser.add_argument("--geometric_weight", type=float, default=0.22)  # Reduced from 0.30
```

---

## Fix 3: Improve Angle Wrapping

### Problem
Angles may be wrapping incorrectly, causing invalid structures and clashes.

### Solution
1. **Verify wrapping happens after each ODE step** (already done)
2. **Add explicit check** that angles are in valid range before structure conversion
3. **Add validation** to reject structures with out-of-range angles

### Code Changes
```python
# In bin/sample_advanced_flow.py, after angle wrapping
# ADD: Validate angles are in correct range
phi = sample_corrected_np[:, 0]
psi = sample_corrected_np[:, 1]
omega = sample_corrected_np[:, 2]

# Check if angles are in valid range
if np.any(np.abs(phi) > np.pi) or np.any(np.abs(psi) > np.pi):
    logger.warning(f"⚠️  Angles out of range! phi: {np.min(phi):.3f} to {np.max(phi):.3f}, psi: {np.min(psi):.3f} to {np.max(psi):.3f}")
    # Re-wrap if needed
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    psi = np.arctan2(np.sin(psi), np.cos(psi))
    sample_corrected_np[:, 0] = phi
    sample_corrected_np[:, 1] = psi

# Verify omega is π
if not np.allclose(omega, np.pi, atol=0.01):
    logger.warning(f"⚠️  Omega not π! Values: {np.min(omega):.3f} to {np.max(omega):.3f}")
    sample_corrected_np[:, 2] = np.pi
```

---

## Fix 4: Add Structure Validation

### Problem
Structures with high clash rates are being generated without validation.

### Solution
Add validation to reject structures with high clash rates during sampling.

### Code Changes
```python
# In bin/sample_advanced_flow.py, after structure conversion
# ADD: Validate structure quality
from foldingdiff.geometric_validation import validate_structure_quality

quality = validate_structure_quality(pdb_file)
if quality['clash_rate'] > 0.5:  # Reject if clash rate > 50%
    logger.warning(f"⚠️  Sample {i} rejected: clash rate {quality['clash_rate']:.3f} > 0.5")
    # Optionally resample
    continue
```

---

## Fix 5: Adjust Training Hyperparameters

### Problem
Training may not be converging properly with current hyperparameters.

### Solution
1. Reduce geometric loss weight (0.30 → 0.22)
2. Consider reducing dropout (0.1 → 0.05) if overfitting
3. Monitor training loss more carefully

### Code Changes
```python
# In bin/train_advanced_flow.py
parser.add_argument("--geometric_weight", type=float, default=0.22)  # Reduced
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.05)  # Reduced
parser.add_argument("--hidden_dropout_prob", type=float, default=0.05)  # Reduced
```

---

## Implementation Order

### Priority 1 (Immediate):
1. ✅ Fix omega validation and logging
2. ✅ Reduce geometric loss weight
3. ✅ Add angle validation

### Priority 2 (After Priority 1):
4. ✅ Add structure validation
5. ✅ Adjust training hyperparameters

---

## Expected Improvements

### After Fixes:
- **Clash Rate:** 0.804 → 0.25-0.35 (2.3-3.2x improvement)
- **Trans Peptide:** 82.7% → >95% (fix)
- **Quality Score:** 0.141 → 0.30-0.45 (2-3x improvement)
- **Ramachandran:** 35.1% → 50-65% (1.4-1.9x improvement)

---

## Testing

After implementing fixes:
1. Re-train model with new hyperparameters
2. Re-sample structures
3. Re-evaluate
4. Verify improvements



