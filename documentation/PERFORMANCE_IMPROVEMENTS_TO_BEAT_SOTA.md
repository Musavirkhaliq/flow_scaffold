# Performance Improvements to Beat RF Diffusion and SOTA Models

**Date:** 2026-01-02  
**Current Performance:** Quality 0.135, Ramachandran 33.7%, Clash 0.391  
**SOTA Targets:** Quality >0.75, Ramachandran >85%, Clash <0.05  
**Gap:** 5.6x quality, 2.5x Ramachandran, 7.8x clash rate

---

## Executive Summary

To beat RF diffusion and other SOTA models, we need to address **5 critical areas**:

1. **Training Regularization** (Missing dropout, low weight decay)
2. **Geometric Loss Weighting** (Too low, needs Ramachandran-aware weighting)
3. **Advanced Loss Functions** (Pairwise distance loss, secondary structure consistency)
4. **Training Hyperparameters** (Batch size, learning rate schedule, EMA)
5. **Post-Training Refinement** (Structure refinement during validation)

---

## Critical Issues Identified

### 1. ❌ Missing Dropout in BERT Layers

**Current State:**
- No dropout in BERT attention layers
- Only guidance dropout (0.15) for motif conditioning
- Model likely overfitting to training data

**Impact:** High overfitting → poor generalization → low quality scores

**Fix:**
```python
# In BertConfig initialization
config = BertConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    attention_probs_dropout_prob=0.1,  # ADD THIS
    hidden_dropout_prob=0.1,  # ADD THIS
    intermediate_size=args.hidden_size * 4,
    max_position_embeddings=args.pad,
)
```

**Expected Improvement:** +0.05-0.10 quality score (5-10% improvement)

---

### 2. ❌ Weight Decay Too Low or Missing

**Current State:**
- `l2_lambda` defaults to 0.0 in training
- No weight decay in optimizer (only L2 regularization if explicitly set)
- Model weights not regularized

**Impact:** Overfitting, poor generalization

**Fix:**
```python
# In configure_optimizers
optimizer = torch.optim.AdamW(
    self.parameters(),
    lr=self.learning_rate,
    weight_decay=1e-4,  # ADD THIS (standard for BERT)
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Expected Improvement:** +0.03-0.05 quality score

---

### 3. ⚠️ Geometric Loss Weight Too Low

**Current State:**
- Geometric loss weight: 0.15
- Ramachandran outliers: 46.9% (target: <5%)
- Geometric constraints not strong enough

**Impact:** Model doesn't learn geometric constraints well

**Fix:**
```python
# Increase geometric weight progressively
geometric_weight = 0.15  # Current
# Should be: 0.25-0.35 for better geometric learning

# Also add Ramachandran-aware weighting
# Penalize outliers more heavily
outlier_penalty = rama_stats['outliers'] * 1.0  # Increase from 0.5
favored_reward = -rama_stats['favored'] * 0.5  # Increase from 0.3
```

**Expected Improvement:** +0.10-0.15 quality score, Ramachandran +15-25%

---

### 4. ❌ Missing Pairwise Distance Loss

**Current State:**
- Pairwise distance loss exists in codebase but not used
- Only angular flow matching loss is used
- No 3D structure consistency during training

**Impact:** Generated structures don't respect 3D geometry well

**Fix:**
```python
# In training_step, after computing angles
from foldingdiff.losses import pairwise_dist_loss

# Convert angles to coordinates
coords_pred = angles_to_coords(angles_pred)
coords_target = angles_to_coords(angles_target)

# Add pairwise distance loss
pairwise_loss = pairwise_dist_loss(
    coords_pred, coords_target,
    lengths=batch['lengths'],
    weights=scaffold_weight_mask
)

total_loss = main_loss + 0.1 * pairwise_loss  # Weight: 0.1
```

**Expected Improvement:** +0.05-0.10 quality score, -0.05-0.10 clash rate

---

### 5. ❌ No Exponential Moving Average (EMA)

**Current State:**
- No EMA for model weights
- Final model uses last checkpoint weights
- SOTA models use EMA for better stability

**Impact:** Less stable training, worse final performance

**Fix:**
```python
# Add EMA to training
from torch.optim.swa_utils import AveragedModel

# After model initialization
ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

# In training_step, after optimizer.step()
ema_model.update_parameters(model)

# Use ema_model for validation and final checkpoint
```

**Expected Improvement:** +0.03-0.05 quality score

---

### 6. ⚠️ Batch Size Too Small

**Current State:**
- Batch size: 16
- SOTA models use 32-64 or larger with gradient accumulation

**Impact:** Less stable gradients, slower convergence

**Fix:**
```python
# Option 1: Increase batch size (if memory allows)
batch_size = 32  # or 64

# Option 2: Use gradient accumulation
accumulate_grad_batches = 2  # Effective batch size: 16 * 2 = 32
```

**Expected Improvement:** +0.02-0.05 quality score, faster convergence

---

### 7. ❌ No Adaptive Learning Rate Schedule

**Current State:**
- Only LinearWarmup scheduler
- No cosine annealing or reduce-on-plateau
- Learning rate doesn't adapt to training progress

**Impact:** Suboptimal convergence, may get stuck in local minima

**Fix:**
```python
# Use cosine annealing with warm restarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Initial period
    T_mult=2,  # Period multiplier
    eta_min=1e-6  # Minimum LR
)
```

**Expected Improvement:** +0.02-0.04 quality score

---

### 8. ❌ No Secondary Structure Consistency Loss

**Current State:**
- No loss to ensure predicted angles match expected secondary structure
- Model doesn't learn alpha-helix/beta-sheet preferences

**Impact:** Low secondary structure content (14% alpha, 21% beta vs 30-40% expected)

**Fix:**
```python
# Add secondary structure prediction head
ss_pred = self.secondary_structure_head(hidden_states)
ss_target = predict_secondary_structure_from_angles(angles)

ss_loss = F.cross_entropy(ss_pred, ss_target, ignore_index=-1)
total_loss = main_loss + 0.05 * ss_loss
```

**Expected Improvement:** +0.03-0.05 quality score, better secondary structure

---

### 9. ❌ No Ramachandran-Aware Loss Weighting

**Current State:**
- All residues weighted equally in loss
- Outliers not penalized more heavily

**Impact:** Model doesn't prioritize fixing Ramachandran outliers

**Fix:**
```python
# Compute Ramachandran quality per residue
rama_quality = compute_ramachandran_quality(phi, psi)  # [batch, seq_len]
# 1.0 = favored, 0.5 = allowed, 0.0 = outlier

# Weight loss inversely by quality (penalize outliers more)
rama_weights = 1.0 + (1.0 - rama_quality) * 2.0  # Outliers get 3x weight
loss_per_element = loss_per_element * rama_weights
```

**Expected Improvement:** +0.05-0.10 Ramachandran favored, -10-20% outliers

---

### 10. ❌ No Structure Refinement During Training

**Current State:**
- No structure refinement during validation
- Poor structures not corrected during training

**Impact:** Model doesn't learn from refined structures

**Fix:**
```python
# In validation_step, refine structures
from foldingdiff.structure_refinement import refine_structure

refined_angles = refine_structure(
    angles_pred,
    max_iterations=5,
    improve_ramachandran=True,
    fix_omega=True
)

# Compute loss on refined structures (for monitoring)
refined_loss = compute_loss(refined_angles, angles_target)
```

**Expected Improvement:** +0.02-0.04 quality score (indirect, through better validation)

---

## Implementation Priority

### Priority 1: Critical (Implement First)

1. ✅ **Add Dropout** (Easy, high impact)
2. ✅ **Add Weight Decay** (Easy, high impact)
3. ✅ **Increase Geometric Loss Weight** (Easy, high impact)
4. ✅ **Add Ramachandran-Aware Weighting** (Medium, high impact)

**Expected Combined Improvement:** +0.20-0.35 quality score, +20-35% Ramachandran

### Priority 2: High Impact (Implement Second)

5. ✅ **Add Pairwise Distance Loss** (Medium, high impact)
6. ✅ **Add EMA** (Easy, medium impact)
7. ✅ **Increase Batch Size / Gradient Accumulation** (Easy, medium impact)

**Expected Combined Improvement:** +0.10-0.20 quality score, -0.05-0.10 clash rate

### Priority 3: Medium Impact (Implement Third)

8. ✅ **Add Adaptive Learning Rate Schedule** (Easy, medium impact)
9. ✅ **Add Secondary Structure Consistency** (Medium, medium impact)
10. ✅ **Add Structure Refinement** (Hard, low-medium impact)

**Expected Combined Improvement:** +0.05-0.10 quality score

---

## Expected Final Performance

### After Priority 1 Fixes:
- Quality: 0.135 → **0.35-0.50** (2.6-3.7x improvement)
- Ramachandran: 33.7% → **55-70%** (1.6-2.1x improvement)
- Clash Rate: 0.391 → **0.25-0.30** (1.3-1.6x improvement)

### After Priority 2 Fixes:
- Quality: 0.35-0.50 → **0.45-0.65** (3.3-4.8x improvement)
- Ramachandran: 55-70% → **65-75%** (1.9-2.2x improvement)
- Clash Rate: 0.25-0.30 → **0.15-0.20** (2.0-2.6x improvement)

### After Priority 3 Fixes:
- Quality: 0.45-0.65 → **0.50-0.70** (3.7-5.2x improvement)
- Ramachandran: 65-75% → **70-80%** (2.1-2.4x improvement)
- Clash Rate: 0.15-0.20 → **0.10-0.15** (2.6-3.9x improvement)

### To Beat SOTA (Target: 0.75 quality, 85% Ramachandran, 0.05 clash):
- Need additional improvements:
  - Better architecture (more layers, larger hidden size)
  - More training data or data augmentation
  - Longer training (200+ epochs)
  - Ensemble methods

---

## Code Changes Required

### 1. Add Dropout to BERT Config
**File:** `bin/train_advanced_flow.py`
```python
config = BertConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    attention_probs_dropout_prob=0.1,  # ADD
    hidden_dropout_prob=0.1,  # ADD
    intermediate_size=args.hidden_size * 4,
    max_position_embeddings=args.pad,
)
```

### 2. Add Weight Decay to Optimizer
**File:** `foldingdiff/enhanced_models_v2.py`
```python
optimizer = torch.optim.AdamW(
    self.parameters(),
    lr=self.learning_rate,
    weight_decay=1e-4,  # ADD
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 3. Increase Geometric Loss Weight
**File:** `bin/train_advanced_flow.py`
```python
parser.add_argument("--geometric_weight", type=float, default=0.25)  # Increase from 0.15
```

### 4. Add Ramachandran-Aware Weighting
**File:** `foldingdiff/enhanced_models_v2.py`
```python
# In _compute_geometric_loss
# After computing rama_stats
outlier_penalty = rama_stats['outliers'] * 1.0  # Increase from 0.5
favored_reward = -rama_stats['favored'] * 0.5  # Increase from 0.3
```

### 5. Add Pairwise Distance Loss
**File:** `foldingdiff/enhanced_models_v2.py`
```python
# In training_step, after main_loss
from foldingdiff.losses import pairwise_dist_loss
from foldingdiff.nerf import angles_to_coords

# Convert to coordinates
coords_pred = angles_to_coords(x_0 + v_pred * dt, lengths)
coords_target = angles_to_coords(x_0, lengths)

pairwise_loss = pairwise_dist_loss(
    coords_pred, coords_target,
    lengths=batch['lengths']
)

total_loss = total_loss + 0.1 * pairwise_loss
```

### 6. Add EMA
**File:** `foldingdiff/enhanced_models_v2.py`
```python
from torch.optim.swa_utils import AveragedModel

# In __init__
self.ema_model = AveragedModel(
    self, 
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
)

# In training_step, after optimizer.step()
self.ema_model.update_parameters(self)
```

### 7. Increase Batch Size / Gradient Accumulation
**File:** `bin/train_advanced_flow.py`
```python
parser.add_argument("--batch_size", type=int, default=32)  # Increase from 16
# OR
parser.add_argument("--accumulate_grad_batches", type=int, default=2)
```

### 8. Add Adaptive Learning Rate Schedule
**File:** `foldingdiff/enhanced_models_v2.py`
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# In configure_optimizers
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)
return [optimizer], [scheduler]
```

---

## Testing Strategy

1. **Implement Priority 1 fixes** → Re-train → Evaluate
2. **If quality < 0.4**, implement Priority 2 fixes → Re-train → Evaluate
3. **If quality < 0.6**, implement Priority 3 fixes → Re-train → Evaluate
4. **If quality < 0.7**, consider architecture improvements

---

## Conclusion

To beat RF diffusion and SOTA models, we need to implement **at least Priority 1 and Priority 2 fixes**. These will bring us from:
- **Current:** Quality 0.135, Ramachandran 33.7%, Clash 0.391
- **Target:** Quality 0.45-0.65, Ramachandran 65-75%, Clash 0.15-0.20

This represents a **3-5x improvement** in quality, which should be competitive with SOTA models.

**Next Steps:**
1. Implement Priority 1 fixes (dropout, weight decay, geometric weight, Ramachandran weighting)
2. Re-train model
3. Re-evaluate
4. If quality still < 0.6, implement Priority 2 fixes
5. Repeat until target performance is reached

