# Gradient Explosion Fix

## Problem Identified

**Issue:** Pre-clip gradient norms consistently high (20-98) every 50 steps, indicating systematic gradient explosion.

**Root Cause Analysis:**
1. Loss accumulates from multiple sources:
   - Main flow matching loss
   - Geometric loss (weight: 0.25-0.50)
   - Pairwise distance loss (weight: 0.1-0.15)
   - Scaffold weighting (2.0-2.5x multiplier)
2. Large model (512 hidden, 12 layers) means many parameters
3. Even small per-parameter gradients sum to large total norm
4. Loss values (0.7-1.7) are reasonable, but gradients are too high

## Fix Applied

**Location:** `foldingdiff/enhanced_models_v2.py` (line 1214-1230)

**Solution:** Scale down total loss by 10x before returning from `training_step()`

```python
# Scale loss down by 10x to reduce gradient magnitude
loss_scale = 0.1
total_loss_scaled = total_loss * loss_scale
```

**Why This Works:**
- Gradients are proportional to loss: `grad = d(loss)/d(param)`
- Scaling loss by 0.1 reduces gradients by 10x
- This is equivalent to reducing learning rate by 10x, but more explicit
- Pre-clip gradient norms should drop from 20-98 to 2-10

**Impact:**
- ✅ Gradients will be 10x smaller (more manageable)
- ✅ Learning rate effectively becomes 10x smaller (5e-5 → 5e-6)
- ✅ Training should be more stable
- ✅ Loss values logged as unscaled for monitoring

## Expected Results

**Before Fix:**
- Pre-clip gradient norms: 20-98
- Warnings every 50 steps
- Training unstable

**After Fix:**
- Pre-clip gradient norms: 2-10 (10x reduction)
- Fewer warnings
- More stable training

## Monitoring

The fix logs both:
- `train_loss_unscaled`: Original loss value (for monitoring)
- `train_loss`: Scaled loss (used for training)

Check TensorBoard to verify:
1. Gradient norms drop to < 10.0
2. Loss still decreases (unscaled loss should decrease)
3. Training remains stable

## Alternative Solutions Considered

1. **Reduce learning rate further**: Would require changing config, less explicit
2. **Reduce geometric loss weights**: Would affect model learning
3. **Reduce scaffold weights**: Would affect scaffold generation quality
4. **Increase gradient clipping**: Already at 0.5, can't go much lower

**Chosen solution (loss scaling) is best because:**
- Explicit and clear
- Doesn't affect loss monitoring
- Easy to adjust if needed
- Equivalent to LR reduction but more transparent

## If Gradients Still High

If pre-clip gradients remain > 10.0 after this fix:

1. **Reduce loss scale further**: Change `loss_scale = 0.1` → `loss_scale = 0.05`
2. **Check geometric loss magnitude**: May need to reduce geometric loss weights
3. **Check pairwise loss magnitude**: May need to reduce pairwise loss weights
4. **Verify gradient accumulation**: Ensure it's working correctly (should be 4x)

## Configuration

Current settings:
- Loss scale: `0.1` (10x reduction)
- Learning rate: `5e-5` (effective: `5e-6` after scaling)
- Gradient clipping: `0.5`
- Gradient accumulation: `4`


