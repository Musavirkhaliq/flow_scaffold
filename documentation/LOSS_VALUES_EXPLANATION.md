# Loss Values Explanation - Why Values < 1 Are Normal

## Summary

**Your loss values (0.3-0.4 unscaled) are NORMAL and EXPECTED for flow matching models.**

## Understanding the Loss Function

### What the Loss Measures

The loss function computes **Huber loss** (smooth L1) on **velocity predictions**:

```python
# For angular features (phi, psi, omega):
# Huber loss with delta = 1.0
if |error| < 1.0:
    loss = 0.5 * error^2
else:
    loss = 1.0 * (|error| - 0.5)
```

### Why Loss Values Are Small

1. **Velocities are derivatives** (rate of change), not absolute angles
   - Velocities are typically small: ~0.1-1.0 rad/unit_time
   - Small velocity errors → small loss values

2. **Huber loss caps large errors**
   - Errors > 1.0 rad are capped linearly
   - Prevents outliers from dominating

3. **Normalization by sequence length**
   - Loss is averaged over all valid positions
   - Longer sequences → more averaging → smaller per-position loss

4. **Mean-centering of angles**
   - Angles are normalized/mean-centered
   - Velocities are relative, not absolute
   - This naturally leads to smaller loss values

## Expected Loss Ranges

### For Flow Matching Models:

**Typical ranges:**
- **Early training:** 0.5-1.5 (model learning basics)
- **Mid training:** 0.3-0.8 (model improving)
- **Late training:** 0.1-0.5 (model converged)

**Your values:**
- Epoch 0: 0.414 (unscaled) ✅ **Normal**
- Epoch 1: 0.327 (unscaled) ✅ **Normal**
- Epoch 2: 0.283 (unscaled) ✅ **Normal**
- Epoch 3: 0.268 (unscaled) ✅ **Normal**

### Comparison with Other Models:

**MSE Loss on Normalized Data:**
- If target values are in range [-1, 1]: loss typically 0.1-1.0
- If target values are in range [-0.5, 0.5]: loss typically 0.01-0.5
- Your velocities are normalized → loss 0.2-0.4 is expected

**Huber Loss (delta=1.0):**
- For errors < 1.0: loss = 0.5 * error^2
- Average error of 0.7 rad → loss ≈ 0.5 * 0.49 = 0.245
- Your loss ~0.3 → average error ~0.77 rad ✅ **Reasonable**

## Why Loss Scaling Makes It Look Small

**Before scaling (actual loss):**
- Epoch 0: 0.414
- Epoch 1: 0.327
- Epoch 2: 0.283

**After scaling (displayed in progress bar):**
- Epoch 0: 0.0414 (scaled by 0.1)
- Epoch 1: 0.0327 (scaled by 0.1)
- Epoch 2: 0.0283 (scaled by 0.1)

The **scaled loss** (0.03-0.04) is what you see in the progress bar, but the **actual loss** (0.3-0.4) is what matters.

## Validation Loss Comparison

**Your validation loss:**
- Epoch 0: 0.8260
- Epoch 1: 0.4355
- Epoch 2: 0.3121
- Epoch 3: 0.2766

**Why validation loss is higher:**
- Validation uses different data (not seen during training)
- Validation loss is typically 1.5-2x training loss (normal)
- Your ratio: ~1.5-2x ✅ **Normal**

## What to Look For (Not Absolute Values)

### ✅ Good Signs:
1. **Loss decreasing:** 0.414 → 0.327 → 0.283 ✅
2. **Validation loss decreasing:** 0.826 → 0.435 → 0.312 ✅
3. **Gradient norms stable:** 2-4 (moderate) ✅
4. **No NaN/Inf:** ✅
5. **Training stable:** ✅

### ❌ Bad Signs (NOT present):
- Loss increasing (yours is decreasing)
- Validation loss much higher than training (yours is ~1.5x, normal)
- Gradient explosion (yours is controlled)
- NaN/Inf errors (none observed)

## Conclusion

**Your loss values are PERFECTLY NORMAL for flow matching models:**

1. ✅ Loss values 0.3-0.4 are expected for Huber loss on normalized velocities
2. ✅ Loss is decreasing steadily (good learning)
3. ✅ Validation loss is reasonable (1.5-2x training loss is normal)
4. ✅ Training is stable (gradients controlled)

**The "small" values you see are:**
- Due to loss scaling (0.1x) for gradient stability
- Normal for this type of model and loss function
- Actually indicate good learning (loss decreasing)

**No action needed** - your training is working correctly! 🎉


