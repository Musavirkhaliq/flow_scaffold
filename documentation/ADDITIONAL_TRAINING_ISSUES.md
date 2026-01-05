# Additional Training Issues Analysis

**Date:** 2026-01-02  
**Focus:** Identify additional training issues beyond geometric loss

---

## Critical Issues Found

### 1. ❌ **GEOMETRIC LOSS BREAKS GRADIENT FLOW** (CRITICAL)

**Location:** `foldingdiff/enhanced_models_v2.py` (lines 1218-1247)

**Problem:**
The geometric loss converts tensors to numpy for Ramachandran checking, which **breaks the computation graph**:

```python
# Convert to numpy for Ramachandran check
phi_np = phi.cpu().numpy()  # ❌ BREAKS GRADIENT FLOW!
psi_np = psi.cpu().numpy()  # ❌ BREAKS GRADIENT FLOW!
mask_np = attention_mask.cpu().numpy()

# Check Ramachandran quality
rama_stats = check_ramachandran(phi_b, psi_b)  # Uses numpy

# Convert back to tensor
rama_loss_tensor = torch.tensor(sum(rama_losses) / len(rama_losses), device=device)
```

**Impact:**
- **No gradients flow through geometric loss!**
- Model doesn't learn from Ramachandran constraints
- Geometric loss is essentially a constant penalty, not a learnable signal
- This completely defeats the purpose of adding geometric loss

**Fix:** Implement Ramachandran checking in PyTorch to maintain gradients.

---

### 2. ⚠️ **NO CURRICULUM LEARNING FOR TIME SAMPLING**

**Location:** `foldingdiff/enhanced_models_v2.py` (lines 875-880)

**Problem:**
The advanced model uses importance-weighted time sampling from the start, but the base class (`BertForFlowMatchingTraining`) uses curriculum learning:

```python
# Base class (flow_models.py) - HAS curriculum learning:
if self.train_epoch_counter < self.epochs * 0.2:
    t = torch.rand(batch_size, device=device) * 0.5  # Easy timesteps
elif self.train_epoch_counter < self.epochs * 0.5:
    t = torch.rand(batch_size, device=device) * 0.8 + 0.1  # Medium
else:
    t = self.flow_schedule.sample_time(..., importance_weighting=True)  # Hard

# Advanced model - NO curriculum learning:
t = self.flow_schedule.sample_time(
    batch_size, device,
    importance_weighting=True,  # Always hard!
    alpha=2.0
)
```

**Impact:**
- Model struggles with difficult timesteps early in training
- Slower convergence
- Potentially worse final performance

**Fix:** Add curriculum learning to advanced model.

---

### 3. ⚠️ **LEARNING RATE MAY BE TOO HIGH**

**Location:** `bin/train_advanced_flow.py` (line 86)

**Problem:**
- Learning rate: `3e-4` (0.0003)
- This is quite high for a complex model with many parameters
- Could lead to unstable training or poor convergence

**Comparison:**
- Base model: `5e-5` (0.00005)
- Advanced model: `3e-4` (0.0003) - **6x higher!**

**Impact:**
- Training instability
- Poor convergence
- Loss may not decrease smoothly

**Fix:** Reduce to `1e-4` or `5e-5` for more stable training.

---

### 4. ⚠️ **BATCH SIZE MAY BE TOO SMALL**

**Location:** `bin/train_advanced_flow.py` (line 85)

**Problem:**
- Batch size: `16`
- Effective batch size: `16 * 2 = 32` (with gradient accumulation)
- For a model with 512 hidden size and 12 layers, this might be too small

**Impact:**
- Noisy gradients
- Slower convergence
- Less stable training

**Fix:** Increase batch size to 32-64 if memory allows, or increase gradient accumulation.

---

### 5. ⚠️ **WARMUP STEPS CALCULATION MAY BE WRONG**

**Location:** `foldingdiff/enhanced_models_v2.py` (line 848)

**Problem:**
```python
num_warmup_steps=int(0.1 * self.epochs * self.steps_per_epoch),
num_training_steps=self.epochs * self.steps_per_epoch
```

This looks correct, but let's verify:
- Warmup: 10% of total steps
- Training steps: total steps
- This should be fine, but the warmup ratio might be too short for a complex model

**Impact:**
- Learning rate increases too quickly
- Early training instability

**Fix:** Increase warmup to 15-20% of training steps.

---

### 6. ⚠️ **NO MIXED PRECISION TRAINING**

**Location:** `bin/train_advanced_flow.py` (line 353)

**Problem:**
```python
precision=32,  # Use 32-bit precision to avoid dtype issues
```

**Impact:**
- Slower training (2x slower)
- Higher memory usage
- Can't train larger models

**Fix:** Enable mixed precision (precision=16) if stable.

---

### 7. ⚠️ **GEOMETRIC LOSS WEIGHT MAY BE TOO LOW**

**Location:** `bin/train_advanced_flow.py` (line 66)

**Problem:**
```python
parser.add_argument("--geometric_weight", type=float, default=0.05)
```

**Impact:**
- Geometric loss has very little influence on training
- Model may not learn geometric constraints effectively

**Fix:** Increase to 0.1-0.2 for stronger geometric constraints.

---

## Summary of Issues

### Critical (Must Fix):
1. ❌ **Geometric loss breaks gradient flow** - No learning from geometric constraints

### Important (Should Fix):
2. ⚠️ **No curriculum learning** - Slower convergence
3. ⚠️ **Learning rate too high** - Training instability
4. ⚠️ **Geometric loss weight too low** - Weak geometric constraints

### Moderate (Consider Fixing):
5. ⚠️ **Batch size too small** - Noisy gradients
6. ⚠️ **Warmup too short** - Early instability
7. ⚠️ **No mixed precision** - Slower training

---

## Recommended Fixes

### Priority 1: Fix Geometric Loss Gradient Flow

**Implement PyTorch-based Ramachandran checking:**

```python
def _compute_ramachandran_loss_pytorch(
    self,
    phi: torch.Tensor,
    psi: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute Ramachandran loss in PyTorch (maintains gradients).
    """
    # Define Ramachandran regions (in radians)
    # Alpha-helix: phi ∈ [-2.0, -0.5], psi ∈ [-1.5, 0.5]
    # Beta-sheet: phi ∈ [-2.5, -0.5], psi ∈ [1.0, 2.5]
    # PPII: phi ∈ [-1.5, 0.0], psi ∈ [0.5, 2.0]
    
    # Check if in favored regions (differentiable)
    alpha_favored = (
        (phi > -2.0) & (phi < -0.5) &
        (psi > -1.5) & (psi < 0.5)
    )
    beta_favored = (
        (phi > -2.5) & (phi < -0.5) &
        (psi > 1.0) & (psi < 2.5)
    )
    ppii_favored = (
        (phi > -1.5) & (phi < 0.0) &
        (psi > 0.5) & (psi < 2.0)
    )
    
    favored = (alpha_favored | beta_favored | ppii_favored).float()
    
    # Penalize outliers (1 - favored)
    outliers = (1.0 - favored) * mask
    outlier_fraction = outliers.sum() / (mask.sum() + 1e-8)
    
    # Reward favored regions
    favored_fraction = (favored * mask).sum() / (mask.sum() + 1e-8)
    
    # Loss: penalize outliers, reward favored
    loss = outlier_fraction * 0.5 - favored_fraction * 0.3
    
    return loss
```

### Priority 2: Add Curriculum Learning

```python
# In training_step, replace:
t = self.flow_schedule.sample_time(
    batch_size, device,
    importance_weighting=True,
    alpha=2.0
)

# With:
if self.train_epoch_counter < self.epochs * 0.2:
    # Early: focus on easy timesteps
    t = torch.rand(batch_size, device=device) * 0.5
elif self.train_epoch_counter < self.epochs * 0.5:
    # Mid: gradually introduce full range
    t = torch.rand(batch_size, device=device) * 0.8 + 0.1
else:
    # Late: importance-weighted sampling
    t = self.flow_schedule.sample_time(
        batch_size, device,
        importance_weighting=True,
        alpha=2.0
    )
```

### Priority 3: Adjust Hyperparameters

```python
# In train_advanced_flow.py:
parser.add_argument("--lr", type=float, default=1e-4)  # Reduced from 3e-4
parser.add_argument("--geometric_weight", type=float, default=0.15)  # Increased from 0.05
parser.add_argument("--warmup_ratio", type=float, default=0.15)  # Keep or increase to 0.2
```

---

## Expected Impact

### If We Fix Geometric Loss Gradient Flow:

**Before:**
- Geometric loss is a constant penalty (no gradients)
- Model doesn't learn geometric constraints
- No improvement from geometric loss

**After:**
- Geometric loss provides learnable signal
- Model learns to favor Ramachandran-allowed regions
- Expected improvement: Ramachandran favored 20% → 50-60%

### If We Add Curriculum Learning:

**Before:**
- Model struggles with difficult timesteps early
- Slower convergence

**After:**
- Gradual introduction of difficult timesteps
- Faster convergence
- Better final performance

### If We Adjust Hyperparameters:

**Before:**
- High LR → unstable training
- Low geometric weight → weak constraints

**After:**
- Stable training
- Stronger geometric constraints
- Better convergence

---

## Implementation Priority

1. **Fix geometric loss gradient flow** (CRITICAL - do immediately)
2. **Add curriculum learning** (IMPORTANT - do soon)
3. **Adjust hyperparameters** (IMPORTANT - do soon)
4. **Increase batch size** (MODERATE - if memory allows)
5. **Enable mixed precision** (MODERATE - if stable)

---

## Conclusion

**The most critical issue is that geometric loss breaks gradient flow.**

This means the geometric loss we added is **completely ineffective** - it's just a constant penalty that doesn't help the model learn. We need to fix this immediately by implementing PyTorch-based Ramachandran checking.

**Combined with the other fixes, we should see significant improvement in model performance.**

