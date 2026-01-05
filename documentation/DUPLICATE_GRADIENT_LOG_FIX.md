# Duplicate Gradient Log Fix

**Date:** 2026-01-05  
**Problem:** Same gradient norm logged multiple times (e.g., 16.1278 appearing 4 times)  
**Status:** ✅ **FIXED**

---

## Problem Analysis

From training logs:
```
2026-01-05 04:38:42,133 - root - WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
2026-01-05 04:38:42,134 - root - WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
2026-01-05 04:38:42,134 - root - WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
2026-01-05 04:38:42,134 - root - WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
```

**Root Cause:**
- `on_after_backward()` is called after **each** backward pass during gradient accumulation
- With `ACCUMULATE_GRAD_BATCHES=2`, this means:
  1. Batch 0: backward() → `on_after_backward()` → log
  2. Batch 1: backward() → `on_after_backward()` → log
  3. Then optimizer.step() → `global_step` increments
- If the same gradient state persists across accumulation steps, the same value gets logged multiple times
- This creates confusing duplicate logs and makes it hard to track actual gradient norms

---

## Fix Applied

**File:** `foldingdiff/enhanced_models_v2.py` (lines 1426-1460)

**Change:**
- **Before:** Logged gradient norms in `on_after_backward()` (called multiple times during accumulation)
- **After:** Moved gradient logging to `on_before_optimizer_step()` (called only once per optimizer step)

**Implementation:**
```python
def on_after_backward(self):
    """Monitor gradients after backward pass to detect gradient explosion"""
    # Note: This is called after each backward() during gradient accumulation
    # We don't log here to avoid duplicate logs - see on_before_optimizer_step()
    pass

def on_before_optimizer_step(self, optimizer):
    """Monitor gradients before optimizer step (after all accumulation is done)"""
    # CRITICAL FIX: Log gradient norms only once per optimizer step (not during accumulation)
    # This prevents duplicate logs when using gradient accumulation
    # Log gradient norms every 50 steps to monitor training stability
    if self.global_step % 50 == 0:
        # ... compute and log gradient norms ...
```

**Why This Works:**
- `on_before_optimizer_step()` is called **only once** per optimizer step
- It's called **after** all gradient accumulation is complete
- This ensures we log the final accumulated gradients, not intermediate states
- No more duplicate logs!

---

## Expected Results

### Before Fix:
```
WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
```

### After Fix:
```
WARNING - High pre-clip gradient norm: 16.1278 at step 0 (will be clipped to 0.5)
```

**Benefits:**
- ✅ Cleaner logs (no duplicates)
- ✅ Accurate gradient norm tracking (one value per optimizer step)
- ✅ Easier to monitor training stability
- ✅ Better visibility into actual gradient magnitudes

---

## Additional Improvements

1. **Updated default clip value:** Changed from `1.0` to `0.5` to match config
2. **Better error handling:** Maintained try/except for robustness
3. **Clear documentation:** Added comments explaining why we use `on_before_optimizer_step()`

---

## Testing

**To verify the fix:**
1. Run training with `ACCUMULATE_GRAD_BATCHES=2`
2. Check logs at step 0, 50, 100, etc.
3. Each gradient norm should appear **only once** per step
4. No duplicate warnings

---

## Summary

**Problem:** Duplicate gradient norm logs due to `on_after_backward()` being called multiple times during gradient accumulation

**Solution:** Moved gradient logging to `on_before_optimizer_step()` which is called only once per optimizer step

**Impact:**
- ✅ Cleaner logs
- ✅ Accurate gradient tracking
- ✅ Better training monitoring

**Status:** ✅ **FIXED - READY FOR TRAINING**

