# Quick Fixes Implementation Guide

**Priority:** CRITICAL - Implement these immediately for maximum impact

---

## Fix 1: Reduce Geometric Loss Weight (5 minutes)

**File:** `config_advanced_flow.sh`

**Change:**
```bash
# Line 54: Change from 0.30 to 0.20
GEOMETRIC_WEIGHT=0.20  # Reduced from 0.30 to prevent training instability
```

**Why:** Geometric weight of 0.30 is causing clash rate to double. Reducing to 0.20 should stabilize training.

---

## Fix 2: Verify Omega Fix Order (10 minutes)

**File:** `bin/sample_advanced_flow.py`

**Check around line 1185-1200:**

```python
# CORRECT ORDER:
# 1. Apply mean correction FIRST
sample_corrected_np = apply_means_with_wrapping(
    sample_np,
    training_means,
    is_angular=[True, True, True, False, False, False]
)

# 2. THEN fix omega to π (trans)
omega_current = sample_corrected_np[:, 2]
omega_target = np.pi
omega_far_from_pi = np.abs(omega_current - omega_target) > 0.1
if np.any(omega_far_from_pi):
    sample_corrected_np[omega_far_from_pi, 2] = omega_target
```

**Verify:** Omega fix happens AFTER mean correction, not before.

---

## Fix 3: Add Time-Dependent Geometric Loss Weighting (15 minutes)

**File:** `foldingdiff/enhanced_models_v2.py`

**Find:** `training_step()` method, around where geometric loss is computed

**Add:**
```python
# After computing geometric_loss, before adding to total_loss:
if self.use_geometric_loss:
    try:
        # Time-dependent weighting: more important near t=0 (clean structure)
        t_mean = t.mean().item()
        time_weight = (1.0 - t_mean).clamp(min=0.0, max=1.0)  # 0 at t=1, 1 at t=0
        
        # Apply time-dependent weight
        geometric_loss_weighted = geometric_loss * time_weight * self.geometric_weight
        
        if not torch.isnan(geometric_loss_weighted) and not torch.isinf(geometric_loss_weighted):
            total_loss = total_loss + geometric_loss_weighted
            log_dict['train_geometric_loss'] = geometric_loss
            log_dict['train_geometric_time_weight'] = time_weight
    except Exception as e:
        logging.warning(f"Error computing geometric loss: {e}")
```

**Why:** Prevents gradient conflicts by weighting geometric loss more at low noise (t→0).

---

## Fix 4: Increase Omega Loss Weight (5 minutes)

**File:** `foldingdiff/flow_matching.py`

**Find:** `compute_angular_flow_matching_loss()` function, around line 801

**Change:**
```python
# Line 801: Change from 2.0 to 3.0
feature_weights = [1.0, 1.0, 3.0, 1.0, 1.0, 1.0]  # omega = 3.0 (was 2.0)
```

**Why:** Stronger penalty for omega deviations encourages trans peptide bonds.

---

## Fix 5: Use RK4 Solver (Already in config, verify)

**File:** `config_advanced_flow.sh`

**Verify line 71:**
```bash
SAMPLING_METHOD="rk4"  # Should be "rk4", not "euler"
```

**If not set:** Change to `"rk4"` for better ODE integration accuracy.

---

## Testing After Fixes

1. **Re-train model** with new geometric weight (0.20)
2. **Sample structures** and verify:
   - Omega trans fraction >95%
   - Clash rate <0.60
   - Quality score >0.20

**Expected improvements:**
- Clash rate: 0.804 → 0.50-0.60 (-25-38%)
- Quality score: 0.141 → 0.18-0.22 (+28-56%)
- Trans peptide: 82.7% → >95% (+15%)

---

## Quick Verification Script

```python
# Check omega fix order
import numpy as np

# Simulate mean correction
sample = np.random.randn(100, 6)
means = np.array([0.0, 0.0, np.pi, 1.92, 2.01, 2.01])  # Typical means

# WRONG ORDER (current bug):
sample_wrong = sample.copy()
sample_wrong[:, 2] = np.pi  # Fix omega first
sample_wrong = sample_wrong + means  # Add mean → omega = 2π → wraps to 0 (cis) ❌

# CORRECT ORDER:
sample_correct = sample.copy()
sample_correct = sample_correct + means  # Add mean first
sample_correct[:, 2] = np.pi  # Fix omega after → omega = π (trans) ✅

print(f"Wrong order omega: {np.mean(sample_wrong[:, 2])} (should be ~π)")
print(f"Correct order omega: {np.mean(sample_correct[:, 2])} (should be ~π)")
```

---

**Time to implement all fixes:** ~35 minutes  
**Expected impact:** Significant improvement in all metrics

