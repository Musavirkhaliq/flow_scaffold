# Config Update Recommendations

**Date:** 2026-01-05  
**Status:** ⚠️ **RECOMMENDATIONS FOR CONFIG UPDATES**

---

## Analysis: Config vs Code Implementation

### Current State

**Config (`config_advanced_flow.sh`):**
- `GEOMETRIC_WEIGHT=0.20`

**Code Implementation (`enhanced_models_v2.py`):**
- Uses `self.geometric_weight` (0.20 from config)
- Then scales it: `base_geometric_weight = self.geometric_weight * 0.5` → **0.10**
- Additional loss scaling: 0.2x early → 0.5x late

**Issue:** Config says 0.20, but code actually uses 0.10 (after 0.5x scaling)

---

## Recommendations

### Option 1: Update Config to Match Actual Usage (RECOMMENDED)

**Change:**
```bash
# In config_advanced_flow.sh
GEOMETRIC_WEIGHT=0.10  # Effective weight after 0.5x scaling in code
```

**Pros:**
- Config accurately reflects actual usage
- Clearer for users
- No confusion about effective weight

**Cons:**
- Need to remove 0.5x scaling in code (or keep it for flexibility)

---

### Option 2: Keep Config at 0.20, Document Scaling (CURRENT)

**Keep:**
```bash
GEOMETRIC_WEIGHT=0.20  # Base weight, scaled to 0.10 in code (0.5x)
```

**Pros:**
- Allows easy adjustment of base weight
- Code can scale it dynamically
- More flexible

**Cons:**
- Confusing (config says 0.20 but actual is 0.10)
- Users might not realize scaling happens

---

### Option 3: Remove Scaling, Use Config Directly (ALTERNATIVE)

**Change Code:**
```python
# Remove 0.5x scaling
base_geometric_weight = self.geometric_weight  # Use config value directly
```

**Change Config:**
```bash
GEOMETRIC_WEIGHT=0.10  # Direct weight (no scaling)
```

**Pros:**
- Simplest - config value is actual value
- No hidden scaling

**Cons:**
- Less flexible
- Need to update config if we want to change

---

## Recommendation: Option 1 (Update Config)

**Reason:** Config should reflect actual effective weight for clarity.

**Action:**
1. Update config: `GEOMETRIC_WEIGHT=0.10`
2. Keep code scaling (for future flexibility) OR remove it
3. Document that effective weight is 0.10

---

## Other Config Values - Status Check

### ✅ Already Correct:
- `LR=3e-5` ✅ (matches code)
- `LR_SCHEDULER="ReduceLROnPlateau"` ✅ (matches code)
- `EPOCHS=150` ✅ (matches code)
- `BATCH_SIZE=16` ✅ (matches code)
- `ACCUMULATE_GRAD_BATCHES=2` ✅ (matches code)
- `GRADIENT_CLIP=0.5` ✅ (matches code)

### ⚠️ Needs Documentation:
- `GEOMETRIC_WEIGHT=0.20` - Actually becomes 0.10 in code (0.5x scaling)
- Loss scaling (0.2x → 0.5x) - Not in config (hardcoded in code)

---

## Suggested Config Update

```bash
# Advanced training options (SOTA OPTIMIZED)
USE_MULTISCALE_LOSS=true
USE_CONSISTENCY_LOSS=true
USE_GEOMETRIC_LOSS=true
CONSISTENCY_WEIGHT=0.15  # SOTA: Slightly higher for better sequence-structure alignment
GEOMETRIC_WEIGHT=0.10  # CRITICAL: Effective weight (code applies 0.5x scaling, so 0.20 → 0.10)
                      # Reduced from 0.20 to prevent gradient explosion (gradient norms 44-90)
USE_OAT_FM=false  # NEW: OAT-FM (Optimal Acceleration Transport) - Optional, enable for better flow matching
```

---

## Summary

**Current Issue:**
- Config says `GEOMETRIC_WEIGHT=0.20`
- Code actually uses `0.20 * 0.5 = 0.10`
- This is confusing

**Recommendation:**
- Update config to `GEOMETRIC_WEIGHT=0.10` to match actual usage
- OR document the scaling clearly in config comments
- OR remove scaling in code and use config value directly

**Best Option:** Update config to 0.10 and document that this is the effective weight after scaling.

