# Complete Config Conflicts Analysis

**Date:** 2026-01-05  
**Status:** ✅ **ALL CONFLICTS IN MAIN CODE FIXED**

---

## Summary

Found and fixed **5 critical conflicts** between `config_advanced_flow.sh` and code defaults in `bin/train_advanced_flow.py`.

---

## Conflicts Fixed in `bin/train_advanced_flow.py`

### ✅ All Fixed - Code Defaults Now Match Config

| Parameter | Config Value | Old Default | New Default | Status |
|-----------|-------------|-------------|-------------|--------|
| `--batch_size` | 16 | 32 ❌ | 16 ✅ | Fixed |
| `--epochs` | 150 | 50 ❌ | 150 ✅ | Fixed |
| `--lr_scheduler` | ReduceLROnPlateau | CosineAnnealing ❌ | ReduceLROnPlateau ✅ | Fixed |
| `--geometric_weight` | 0.20 | 0.15 ❌ | 0.20 ✅ | Fixed |
| `--accumulate_grad_batches` | 2 | 1 ❌ | 2 ✅ | Fixed |

---

## Additional Scripts with Different Values

### ⚠️ `train_and_evaluate_advanced_flow.sh` (Separate Script)

This script has its own hardcoded values and **does NOT source** `config_advanced_flow.sh`:

**Conflicts:**
- `EPOCHS=50` (config: 150)
- `LR=5e-4` (config: 3e-5)
- `LR_SCHEDULER="CosineAnnealing"` (config: ReduceLROnPlateau)
- `GEOMETRIC_WEIGHT=0.25` (config: 0.20)
- `GRADIENT_CLIP=1.0` (config: 0.5)

**Status:** ⚠️ **Not a problem** - This is a separate script that doesn't use the config file. It's for a different workflow.

**Recommendation:** If you want to use this script, either:
1. Update it to source `config_advanced_flow.sh`
2. Or update its values to match config
3. Or document that it's a separate workflow

---

### ✅ `phase1_train_advanced_flow.sh` (Main Script)

**Status:** ✅ **No conflicts**
- Sources `config_advanced_flow.sh` correctly
- Passes all config values to training script
- No hardcoded values that override config

---

## Verification Checklist

### ✅ Main Training Pipeline (`phase1_train_advanced_flow.sh`):
- [x] Sources config file
- [x] Passes all config values
- [x] No conflicts

### ✅ Training Script (`bin/train_advanced_flow.py`):
- [x] Defaults match config values
- [x] Accepts config values via arguments
- [x] No conflicts

### ⚠️ Alternative Script (`train_and_evaluate_advanced_flow.sh`):
- [ ] Does NOT source config (intentional - separate workflow)
- [ ] Has different values (intentional - separate workflow)
- [ ] Not a conflict if not used

---

## Impact of Fixes

### Before Fixes:
- If config wasn't sourced, wrong defaults would be used
- Model would train for 50 epochs instead of 150
- Wrong scheduler (CosineAnnealing instead of ReduceLROnPlateau)
- Weaker geometric constraints (0.15 instead of 0.20)

### After Fixes:
- ✅ Code defaults match config values
- ✅ Training will use correct values even if config isn't sourced
- ✅ No conflicts between config and code
- ✅ More robust - works correctly in all scenarios

---

## Files Modified

1. ✅ `bin/train_advanced_flow.py` - Updated 5 default values to match config

---

## Files Verified (No Changes Needed)

1. ✅ `phase1_train_advanced_flow.sh` - Correctly uses config
2. ✅ `config_advanced_flow.sh` - Correct values
3. ⚠️ `train_and_evaluate_advanced_flow.sh` - Separate workflow (not a conflict)

---

## Recommendations

1. ✅ **Use `phase1_train_advanced_flow.sh`** - This is the main script that properly uses config
2. ⚠️ **If using `train_and_evaluate_advanced_flow.sh`** - Be aware it has different values
3. ✅ **Code defaults now match config** - More robust if config isn't sourced

---

**All conflicts in main code fixed!** ✅

The main training pipeline (`phase1_train_advanced_flow.sh` → `bin/train_advanced_flow.py`) is now fully aligned with `config_advanced_flow.sh`.

