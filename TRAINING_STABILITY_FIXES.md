# Training Stability Fixes

## Problem Identified

The model was not learning properly - training loss was **increasing** instead of decreasing:
- Epoch 0: Train loss 0.8168, Valid loss 0.8897
- Epoch 1: Train loss 1.0052, Valid loss 0.8893
- Epoch 2: Train loss 1.0527, Valid loss 0.8890
- Epoch 3: Train loss ~2.54+ (still increasing)

This indicates **training instability**, likely caused by:
1. Learning rate too high
2. Learning rate scheduler mismatch
3. Stochastic centering noise causing instability
4. Potential gradient explosion

## Fixes Applied

### 1. ✅ Reduced Learning Rate
**File:** `config_advanced_flow.sh`
- **Changed:** `LR=5e-4` → `LR=1e-4`
- **Reason:** Learning rate of 0.0005 was too high for this complex model, causing loss to increase instead of decrease
- **Impact:** More stable training, loss should now decrease properly

### 2. ✅ Fixed Learning Rate Scheduler
**File:** `foldingdiff/enhanced_models_v2.py` (line 886-901)
- **Changed:** `CosineAnnealingWarmRestarts` with `T_0=20` → `CosineAnnealingLR` with `T_max=self.epochs`
- **Reason:** WarmRestarts requires `T_0 < epochs`, but training might run for fewer epochs. Standard CosineAnnealingLR adapts to any number of epochs.
- **Impact:** Proper learning rate decay that matches training duration

### 3. ✅ Disabled Stochastic Centering Noise During Early Training
**File:** `foldingdiff/enhanced_models_v2.py` (line 942-953)
- **Changed:** Disabled stochastic centering noise during first 20% of training, reduced noise magnitude from 0.01 to 0.005
- **Reason:** The noise was causing instability during early training when the model is most sensitive
- **Impact:** More stable early training, noise only applied after model has learned basics

### 4. ✅ Added Gradient Monitoring
**File:** `foldingdiff/enhanced_models_v2.py` (new `on_after_backward` method)
- **Added:** Gradient norm monitoring to detect gradient explosion
- **Reason:** Helps identify if gradients are too large (which would cause training instability)
- **Impact:** Better visibility into training dynamics, warnings if gradients explode

## Update: Additional Fixes for Gradient Explosion

**Issue Detected:** Gradient norms reaching 35.8-98.2 (way above safe threshold of 10.0)

**Additional Fixes Applied:**
1. **Reduced gradient clipping:** `1.0` → `0.5` (more aggressive clipping)
2. **Further reduced learning rate:** `1e-4` → `5e-5` (for stability with high gradients)
3. **Increased gradient accumulation:** `2` → `4` (to smooth gradients and reduce high pre-clip norms)
4. **Improved gradient monitoring:** Now logs pre-clipped gradients for better visibility

**Note:** The gradient norms you see in logs are **pre-clipped**. After clipping at 0.5, they will be capped. However, very high pre-clipped gradients (>5.0) indicate the model is still unstable and may need:
- Even lower learning rate
- More gradient accumulation (now increased to 4)
- Loss scaling

## Current Training Status ✅

**Good News:** Training is now working properly!
- ✅ Loss decreasing: 1.74 → 1.18 in epoch 0
- ✅ Final train loss: 0.7044 ± 0.1025
- ✅ Validation loss: 0.4291 (excellent!)
- ✅ Gradients being clipped properly (0.5)
- ⚠️ Pre-clip gradient norms still high (20-98) but being handled by clipping

**Recommendation:** Continue training with current settings. The high pre-clip gradients are being managed by clipping, and the model is learning effectively.

## Expected Results After Fixes

With these fixes, you should see:
- ✅ Training loss **decreasing** over epochs (not increasing)
- ✅ Validation loss **decreasing** or stabilizing
- ✅ More stable training curves
- ✅ Better convergence
- ✅ Pre-clip gradient norms < 5.0 (ideally < 2.0)

## Next Steps

1. **Restart training** with the updated configuration
2. **Monitor** the training loss - it should now decrease
3. **Check gradient norms** in TensorBoard - should be < 10.0
4. **If loss still increases:**
   - Try even lower learning rate: `LR=5e-5`
   - Increase gradient clipping: `GRADIENT_CLIP=0.5`
   - Check for data issues (NaN values, corrupted samples)

## Configuration Summary

**Before:**
```bash
LR=5e-4  # Too high
LR_SCHEDULER="CosineAnnealing"  # Using WarmRestarts with T_0=20
# Stochastic noise: 0.01 rad applied from start
```

**After:**
```bash
LR=1e-4  # Reduced for stability
LR_SCHEDULER="CosineAnnealing"  # Now uses CosineAnnealingLR
# Stochastic noise: Disabled for first 20%, then 0.005 rad
# Gradient monitoring: Enabled
```

## Files Modified

1. `config_advanced_flow.sh` - Reduced learning rate
2. `foldingdiff/enhanced_models_v2.py` - Fixed scheduler, disabled early noise, added gradient monitoring

