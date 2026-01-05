# Root Cause Analysis: Why Model Gives Poor Results

## Executive Summary

The model generates structures with **very poor geometric quality** (19.5% Ramachandran favored vs 80% target) due to **multiple fundamental issues** in the sampling and angle-to-structure conversion pipeline.

---

## Critical Issues Identified

### 1. ❌ **INCORRECT MEAN CORRECTION** (Most Critical)

**Problem:**
The sampling code uses **hardcoded mean values** that may not match the actual training data means:

```python
# Current code (WRONG):
sample_corrected[:, 2] += np.pi  # omega: add 180° for trans
sample_corrected[:, 3] += 1.92  # tau: add ~110°
sample_corrected[:, 4] += 2.01  # CA:C:1N: add ~115°
sample_corrected[:, 5] += 2.11  # C:1N:1CA: add ~121°
```

**Why This Is Wrong:**

1. **Training uses `wrapped_mean`**: The dataset computes means using `wrapped_mean()` which handles angular features correctly (uses `arctan2(mean(sin), mean(cos))`)

2. **Fixed values may be incorrect**: The actual training means are computed from the dataset and stored, but the sampling code ignores them

3. **Angular vs non-angular confusion**: 
   - `phi`, `psi`, `omega` are angular (wrapped mean)
   - `tau`, `CA:C:1N`, `C:1N:1CA` are bond angles (regular mean, but still need correct values)

**Impact:** If means are wrong, all angles are shifted, leading to geometrically invalid structures.

**Solution:** Load actual training means from `training_mean_offset.npy` or compute from dataset.

---

### 2. ❌ **NO ANGLE WRAPPING DURING SAMPLING**

**Problem:**
Flow matching sampling may not properly wrap angular features to `[-π, π]` during the ODE integration steps.

**Current Flow:**
```python
# In flow_sampling.py
x_next = x_t - dt * v_t  # Euler step
# No wrapping of angular features!
```

**Why This Is Wrong:**

1. **Angular features can go out of range**: Without wrapping, `phi`, `psi`, `omega` can exceed `[-π, π]`
2. **NERF expects angles in `[-π, π]`**: The NERF algorithm assumes angles are in this range
3. **Out-of-range angles cause geometric errors**: Invalid angles lead to invalid structures

**Impact:** Angles outside `[-π, π]` cause NERF to generate geometrically invalid coordinates.

**Solution:** Wrap angular features after each ODE step:
```python
if is_angular[j]:
    x_next[:, :, j] = utils.modulo_with_wrapped_range(
        x_next[:, :, j], range_min=-np.pi, range_max=np.pi
    )
```

---

### 3. ❌ **NO GEOMETRIC CONSTRAINTS DURING SAMPLING**

**Problem:**
The model samples angles without enforcing geometric constraints (Ramachandran, bond angles, etc.).

**Why This Is Wrong:**

1. **Model learns distribution, not constraints**: The model learns the distribution of angles but doesn't enforce physical constraints
2. **Sampling can violate constraints**: Even if training data is valid, sampling can produce invalid angles
3. **No validation during sampling**: Angles are not checked for validity until after structure generation

**Impact:** Many sampled angles violate geometric constraints, leading to poor structures.

**Solution:** 
- Add geometric validation during sampling
- Reject invalid samples (already implemented)
- Add constraint penalties to loss function (training)

---

### 4. ❌ **BOND ANGLES NOT PROPERLY CONSTRAINED**

**Problem:**
Bond angles (`tau`, `CA:C:1N`, `C:1N:1CA`) have physical ranges but are not constrained:

- `tau` (N-CA-C): Should be ~110° (1.92 rad), range ~100-130° (1.75-2.27 rad)
- `CA:C:1N`: Should be ~115° (2.01 rad), range ~105-125° (1.83-2.18 rad)
- `C:1N:1CA`: Should be ~121° (2.11 rad), range ~110-130° (1.92-2.27 rad)

**Current:** These are sampled as if they can be any value.

**Impact:** Invalid bond angles cause incorrect bond geometry, leading to clashes and poor structures.

**Solution:** Constrain bond angles to valid ranges during sampling.

---

### 5. ⚠️ **TRAINING DATA QUALITY UNKNOWN**

**Problem:**
We don't know if the training data itself has geometric issues.

**Questions:**
- Are CATH structures geometrically valid?
- Are there preprocessing errors?
- Is the angle extraction correct?

**Impact:** If training data is poor, model learns poor distributions.

**Solution:** Validate training data quality.

---

### 6. ⚠️ **NERF CONVERSION MAY HAVE ISSUES**

**Problem:**
The NERF algorithm converts angles to coordinates, but:
- Initial coordinates are fixed (may not be optimal)
- Bond lengths are fixed (may not match actual values)
- Error propagation through chain

**Impact:** Even with correct angles, NERF may produce invalid structures.

**Solution:** Validate NERF conversion with known-good angles.

---

## Detailed Analysis

### Mean Correction Issue

**Training:**
```python
# In datasets.py
self.means = cm.wrapped_mean(structures_concat, axis=0)  # Computed from data
angles = angles - self.means  # Subtract means during training
```

**Sampling (WRONG):**
```python
# In sample_advanced_flow.py
sample_corrected[:, 2] += np.pi  # Hardcoded!
sample_corrected[:, 3] += 1.92  # Hardcoded!
```

**Correct Approach:**
```python
# Should load from training_mean_offset.npy or compute from dataset
means = train_dset.dset.get_masked_means()  # Actual training means
sample_corrected = sample + means  # Add actual means
```

### Angle Wrapping Issue

**Current Flow Matching:**
```python
# flow_sampling.py - NO WRAPPING!
x_next = x_t - dt * v_t
```

**Should Be:**
```python
x_next = x_t - dt * v_t
# Wrap angular features
for j in range(n_features):
    if is_angular[j]:
        x_next[:, :, j] = utils.modulo_with_wrapped_range(
            x_next[:, :, j], range_min=-np.pi, range_max=np.pi
        )
```

### Geometric Constraints

**Missing:**
- Ramachandran validation during sampling
- Bond angle range constraints
- Clash detection during sampling
- Geometric loss in training

---

## Root Cause Priority

### Priority 1: CRITICAL (Fix Immediately)

1. **Mean Correction** - Using wrong means shifts all angles
2. **Angle Wrapping** - Out-of-range angles break NERF
3. **Bond Angle Constraints** - Invalid bond angles cause clashes

### Priority 2: HIGH (Fix Soon)

4. **Geometric Constraints During Sampling** - Need validation/rejection
5. **Training Data Validation** - Ensure training data is valid

### Priority 3: MODERATE (Fix Later)

6. **NERF Validation** - Verify NERF conversion
7. **Advanced Refinement** - Post-processing improvements

---

## Expected Impact of Fixes

### If We Fix Mean Correction + Angle Wrapping:

**Before:**
- Quality: 0.105
- Ramachandran Favored: 19.5%
- Outliers: 66.3%

**After (Expected):**
- Quality: 0.3-0.4 (3-4x improvement)
- Ramachandran Favored: 40-50% (2x improvement)
- Outliers: 40-50% (reduced)

### If We Also Add Geometric Constraints:

**After (Expected):**
- Quality: 0.5-0.6
- Ramachandran Favored: 60-70%
- Outliers: 20-30%

---

## Recommended Fixes

### Immediate (This Week)

1. **Fix Mean Correction:**
   ```python
   # Load actual training means
   means = np.load("results/advanced_flow/model/training_mean_offset.npy")
   sample_corrected = sample + means
   ```

2. **Add Angle Wrapping:**
   ```python
   # Wrap after each ODE step
   is_angular = [True, True, True, False, False, False]  # phi, psi, omega, tau, CA:C:1N, C:1N:1CA
   for j in range(6):
       if is_angular[j]:
           x_next[:, :, j] = utils.modulo_with_wrapped_range(
               x_next[:, :, j], -np.pi, np.pi
           )
   ```

3. **Add Bond Angle Constraints:**
   ```python
   # Clamp bond angles to valid ranges
   sample_corrected[:, 3] = np.clip(sample_corrected[:, 3], 1.75, 2.27)  # tau
   sample_corrected[:, 4] = np.clip(sample_corrected[:, 4], 1.83, 2.18)  # CA:C:1N
   sample_corrected[:, 5] = np.clip(sample_corrected[:, 5], 1.92, 2.27)  # C:1N:1CA
   ```

### Short-Term (Next 2 Weeks)

4. **Add Geometric Validation During Sampling** (already implemented, just need to enable)
5. **Validate Training Data Quality**
6. **Add Geometric Loss to Training**

---

## Conclusion

The poor results are caused by **fundamental issues in the sampling pipeline**, not necessarily the model itself:

1. **Wrong mean correction** - Using hardcoded values instead of training means
2. **No angle wrapping** - Angular features go out of range
3. **No geometric constraints** - Invalid angles are not rejected

**The model architecture is likely fine** - the issue is in how angles are processed during sampling.

**Priority:** Fix mean correction and angle wrapping first - these alone should provide 2-3x improvement in quality.

