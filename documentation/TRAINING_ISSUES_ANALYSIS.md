# Training Issues Analysis

**Date:** 2026-01-02  
**Focus:** Identify training issues causing low model performance

---

## Critical Issues Found in Training

### 1. ❌ **ADVANCED LOSSES ARE DISABLED** (CRITICAL)

**Location:** `foldingdiff/enhanced_models_v2.py` (lines 971-986)

**Problem:**
All advanced loss components are **commented out/disabled**:

```python
# Skip advanced losses for now to isolate the issue
# Multi-scale loss (if enabled) - DISABLED FOR DEBUGGING
# if self.use_multiscale_loss and "multiscale" in self.flow_components:
#     ...

# Sequence-structure consistency loss (if enabled) - DISABLED FOR DEBUGGING
# if (self.use_consistency_loss and 
#     self.use_sequence_augmentation and 
#     'sequences' in batch):
#     ...

# Geometric constraint loss (if enabled) - DISABLED FOR DEBUGGING
# if (self.use_geometric_loss and 
#     'motif_coords' in batch and 
#     batch.get('motif_coords') is not None):
#     ...
```

**Impact:**
- **No geometric constraint learning** - Model doesn't learn to avoid invalid geometries
- **No Ramachandran penalty** - Model doesn't learn to favor allowed regions
- **No omega trans penalty** - Model doesn't learn to favor trans peptide bonds
- **No consistency learning** - Model doesn't learn sequence-structure consistency

**Why This Matters:**
- Model only learns basic flow matching loss
- No geometric constraints enforced during training
- Model can learn to generate invalid structures
- This explains why sampling produces poor structures

**Fix:** Re-enable these losses with proper implementation.

---

### 2. ❌ **NO OMEGA-SPECIFIC LOSS** (CRITICAL)

**Problem:**
The loss function treats omega the same as phi/psi, but:
- **Omega should be ~π (trans)** for 95%+ of residues
- **Current training:** Omega is mean-centered, so training data has omega ~0
- **Model learns:** Omega distribution centered at ~0
- **Result:** Model generates cis peptide bonds (omega ~0) instead of trans (omega ~π)

**Current Loss:**
```python
# All angular features treated the same
if angular:
    diff = pred_v - target_v
    loss_per_element = torch.where(
        abs_diff < huber_delta,
        0.5 * diff ** 2,
        huber_delta * (abs_diff - 0.5 * huber_delta)
    )
```

**What's Missing:**
- No penalty for omega being far from π
- No constraint to favor trans peptide bonds
- No special handling for omega in loss function

**Fix:** Add omega-specific loss that penalizes deviations from π (trans).

---

### 3. ⚠️ **LOSS FUNCTION DOESN'T HANDLE OMEGA CORRECTLY**

**Problem:**
The angular loss function uses Huber loss for all angular features, but:
- For omega, we want it to be **close to π (trans)**, not just "in range"
- Current loss treats omega ~0 and omega ~π equally (both are valid angles)
- But omega ~0 (cis) is **biologically invalid** for most residues

**Fix:** Add omega-specific penalty:
```python
# Omega should be close to π (trans)
if i == 2:  # omega is index 2
    # Penalize deviations from π
    omega_penalty = torch.abs(pred_v[:, :, 2] - target_v[:, :, 2])
    # Add extra penalty if target is far from π
    trans_penalty = torch.where(
        torch.abs(target_v[:, :, 2]) > 0.5,  # Far from π
        2.0,  # Higher penalty
        1.0   # Normal penalty
    )
    loss_per_element = loss_per_element * trans_penalty
```

---

### 4. ⚠️ **TRAINING DATA OMEGA DISTRIBUTION**

**Issue:**
- Training data has omega mean ~π (trans)
- After mean centering: omega ~0
- Model learns: omega distribution centered at 0
- During sampling: Model generates omega ~0 (cis)
- After adding mean: omega ~π (should be correct)
- **BUT:** If mean correction is wrong, omega stays at ~0

**This explains the omega issue!**

---

### 5. ⚠️ **NO GEOMETRIC CONSTRAINT LOSS**

**Problem:**
Geometric constraint loss is disabled, so model doesn't learn:
- Ramachandran constraints
- Bond angle constraints
- Clash avoidance
- Secondary structure preferences

**Impact:**
- Model can learn to generate geometrically invalid structures
- No penalty for poor Ramachandran quality
- No penalty for clashes

---

### 6. ⚠️ **NO VALIDATION DURING TRAINING**

**Problem:**
Training doesn't validate:
- Generated structure quality
- Ramachandran statistics
- Clash rates
- Omega trans/cis ratios

**Impact:**
- Can't detect if model is learning poor distributions
- No early warning of geometric issues
- Training may converge to poor local minima

---

## Root Cause: Training Loss Issues

### Primary Issue: **No Geometric Constraints in Loss**

The model is trained with **only basic flow matching loss**, no geometric constraints:

1. **No omega trans penalty** → Model learns omega ~0 (cis) instead of ~π (trans)
2. **No Ramachandran penalty** → Model learns invalid conformations
3. **No clash penalty** → Model learns structures with clashes
4. **No geometric loss** → Model doesn't learn physical constraints

### Secondary Issue: **Mean Centering Confusion**

- Training: omega mean ~π, after centering omega ~0
- Model learns: omega distribution centered at 0
- Sampling: Generates omega ~0
- Mean correction: Should add π back → omega ~π
- **But if mean correction fails, omega stays at 0 (cis)**

---

## Recommended Fixes

### Priority 1: Re-enable Geometric Loss

```python
# In training_step, re-enable geometric loss
if (self.use_geometric_loss and 
    'coords_computed' in batch):
    geometric_loss = self._compute_geometric_loss(
        v_pred, x_0, batch['coords_computed']
    )
    total_loss += self.geometric_weight * geometric_loss
```

**Implementation needed:**
- Ramachandran penalty (favor allowed regions)
- Omega trans penalty (favor omega ~π)
- Clash penalty (penalize clashes)
- Bond angle constraints

### Priority 2: Add Omega-Specific Loss

```python
# In compute_angular_flow_matching_loss
if i == 2:  # omega
    # Add penalty for omega far from π (trans)
    omega_target = target_v[:, :, 2]
    omega_pred = pred_v[:, :, 2]
    
    # Penalize if target is far from π (should be trans)
    trans_penalty = torch.where(
        torch.abs(omega_target) < 1.0,  # Close to 0 (cis) - bad!
        2.0,  # Higher penalty
        1.0   # Normal penalty (close to π - good!)
    )
    
    diff = omega_pred - omega_target
    loss_per_element = torch.where(
        abs_diff < huber_delta,
        0.5 * diff ** 2 * trans_penalty,
        huber_delta * (abs_diff - 0.5 * huber_delta) * trans_penalty
    )
```

### Priority 3: Add Training Validation

```python
# Periodically validate generated structures during training
if batch_idx % 1000 == 0:
    # Sample a few structures
    samples = self.sample(n=5)
    # Validate quality
    quality = validate_structure_quality(samples)
    # Log metrics
    self.log('train_rama_favored', quality['rama_favored'])
    self.log('train_clash_rate', quality['clash_rate'])
    self.log('train_trans_fraction', quality['trans_fraction'])
```

---

## Expected Impact of Training Fixes

### If We Add Geometric Loss:

**Before:**
- Model learns only flow matching
- No geometric constraints
- Poor structure quality

**After:**
- Model learns geometric constraints
- Better Ramachandran quality
- Fewer clashes
- More trans peptide bonds

**Expected Improvement:**
- Quality: 0.081 → 0.4-0.6 (5-7x improvement)
- Ramachandran Favored: 20% → 50-70% (2.5-3.5x improvement)
- Clash Rate: 0.999 → 0.2-0.4 (5x improvement)

### If We Add Omega-Specific Loss:

**Before:**
- Omega ~0 (cis) - 85-98%
- Trans ~π - 1.6-19.2%

**After:**
- Omega ~π (trans) - >95%
- Cis ~0 - <5%

**Expected Improvement:**
- Trans fraction: 12-46% → >95%
- Clash rate: 0.999 → 0.2-0.4 (due to correct geometry)

---

## Implementation Plan

### Immediate (This Week):

1. **Re-enable Geometric Loss** (with proper implementation)
2. **Add Omega-Specific Loss** (penalize cis, favor trans)
3. **Add Training Validation** (monitor quality during training)

### Short-Term (Next 2 Weeks):

4. **Implement Ramachandran Penalty** in loss
5. **Implement Clash Penalty** in loss
6. **Add Bond Angle Constraints** in loss

### Long-Term (Next Month):

7. **Multi-Scale Loss** (re-enable with fixes)
8. **Consistency Loss** (re-enable with fixes)
9. **Advanced Training Strategies** (curriculum learning, etc.)

---

## Conclusion

**The training process has critical issues:**

1. ❌ **Advanced losses disabled** - No geometric constraints learned
2. ❌ **No omega-specific loss** - Model learns cis instead of trans
3. ❌ **No geometric penalty** - Model can learn invalid structures

**These training issues explain why the model generates poor structures.**

**Fixes needed:**
- Re-enable geometric loss (with proper implementation)
- Add omega-specific loss (favor trans)
- Add training validation (monitor quality)

**Combined with sampling fixes, this should dramatically improve quality.**

