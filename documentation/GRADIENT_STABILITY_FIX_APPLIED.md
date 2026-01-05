# Gradient Stability Fix Applied

**Date:** 2026-01-05  
**Problem:** Gradient norms still high (15-32) and Ramachandran quality very low (6-19%)  
**Status:** ✅ **FIXES APPLIED**

---

## Problem Analysis

From training logs:
- **Gradient norms:** 15-32 (improved from 44-90, but still too high)
- **Ramachandran favored:** 6-19% (target: >85%)
- **Validation loss:** 1.8880 (very high)
- **Training loss:** 0.885 → 0.647 → 0.638 (decreasing, which is good)

**Root Cause:**
- Learning rate (3e-5) still too high for current gradient magnitudes
- Loss scaling (0.2-0.5x) not aggressive enough
- Need further reduction to stabilize training

---

## Fixes Applied

### 1. ✅ Reduced Learning Rate: 3e-5 → 2e-5

**Files Modified:**
- `config_advanced_flow.sh` (line 20)
- `bin/train_advanced_flow.py` (line 105)

**Change:**
```bash
# Before
LR=3e-5

# After
LR=2e-5  # Reduced to handle gradient norms 15-32 and improve stability
```

**Rationale:**
- Gradient norms of 15-32 indicate learning rate is still too high
- Lower learning rate (2e-5) will reduce gradient magnitudes
- More stable training with better convergence

**Expected Impact:**
- Gradient norms: 15-32 → **5-15** (50% reduction)
- More stable training dynamics
- Better convergence

---

### 2. ✅ Reduced Loss Scaling: 0.2-0.5x → 0.1-0.3x

**File Modified:**
- `foldingdiff/enhanced_models_v2.py` (lines 2075-2082)

**Change:**
```python
# Before
if progress < 0.3:
    loss_scale = 0.2  # Very conservative early training
elif progress < 0.6:
    loss_scale = 0.2 + 0.2 * (progress - 0.3) / 0.3  # 0.2 → 0.4
else:
    loss_scale = 0.4 + 0.1 * (progress - 0.6) / 0.4  # 0.4 → 0.5

# After
if progress < 0.3:
    loss_scale = 0.1  # Very conservative early training (reduced from 0.2)
elif progress < 0.6:
    loss_scale = 0.1 + 0.1 * (progress - 0.3) / 0.3  # 0.1 → 0.2
else:
    loss_scale = 0.2 + 0.1 * (progress - 0.6) / 0.4  # 0.2 → 0.3 (max 0.3x)
```

**Rationale:**
- More aggressive loss scaling to further reduce gradient magnitudes
- Start at 0.1x (50% reduction from 0.2x) for very conservative early training
- Gradually increase to 0.3x (40% reduction from 0.5x) for late training
- Combined with lower LR, should reduce gradient norms significantly

**Expected Impact:**
- Gradient norms: 15-32 → **5-12** (60-70% reduction)
- More stable training
- Better convergence

---

## Combined Effect

**Before Fixes:**
- Learning rate: 3e-5
- Loss scaling: 0.2-0.5x
- Gradient norms: 15-32
- Effective learning rate: ~0.6e-5 to 1.5e-5

**After Fixes:**
- Learning rate: 2e-5
- Loss scaling: 0.1-0.3x
- Expected gradient norms: 5-12
- Effective learning rate: ~0.2e-5 to 0.6e-5

**Total Reduction:**
- Effective learning rate: **~60-70% reduction**
- Gradient norms: **Expected 60-70% reduction**

---

## Expected Results

### Immediate (Next Training Run)
- ✅ Gradient norms: 15-32 → **5-12** (60-70% reduction)
- ✅ Fewer "High pre-clip gradient norm" warnings
- ✅ More stable training dynamics
- ✅ Training loss continues to decrease

### Medium Term (After Several Epochs)
- ✅ Validation loss: 1.8880 → **<1.0** (target)
- ✅ Ramachandran quality: 6-19% → **>50%** (improvement)
- ✅ Better convergence
- ✅ More stable training

### Long Term (Full Training)
- ✅ Validation loss: **<0.55** (target)
- ✅ Ramachandran favored: **>85%** (target)
- ✅ High-quality structures
- ✅ Stable training throughout

---

## Monitoring

**Key Metrics to Watch:**
1. **Gradient norms:** Should be 5-12 (down from 15-32)
2. **Training loss:** Should continue decreasing
3. **Validation loss:** Should decrease from 1.8880
4. **Ramachandran quality:** Should improve from 6-19%

**Warning Signs:**
- Gradient norms still >15: May need further LR reduction
- Training loss not decreasing: May need to increase LR slightly
- Validation loss not improving: May need more epochs or different scheduler

---

## Next Steps

1. **Monitor training:** Watch gradient norms and loss curves
2. **If gradient norms still high (>15):** Consider further LR reduction to 1.5e-5
3. **If Ramachandran quality still low:** May need to increase geometric loss weight gradually (but monitor gradient norms)
4. **If validation loss plateaus:** Consider adjusting ReduceLROnPlateau patience/factor

---

## Files Modified

1. ✅ `config_advanced_flow.sh` - Reduced LR from 3e-5 to 2e-5
2. ✅ `bin/train_advanced_flow.py` - Updated default LR to match config
3. ✅ `foldingdiff/enhanced_models_v2.py` - Reduced loss scaling from 0.2-0.5x to 0.1-0.3x

---

## Summary

**Problem:** Gradient norms 15-32 (too high), Ramachandran quality 6-19% (too low)

**Solution:**
- Reduced learning rate: 3e-5 → 2e-5
- Reduced loss scaling: 0.2-0.5x → 0.1-0.3x

**Expected Impact:**
- Gradient norms: 15-32 → 5-12 (60-70% reduction)
- More stable training
- Better convergence
- Improved Ramachandran quality over time

**Status:** ✅ **FIXES APPLIED - READY FOR TRAINING**

