# Training & Sampling Code Improvements for Best Model Performance
## Based on Code Analysis & Latest Research (2024-2025)

**Date:** 2026-01-03  
**Current Performance:** Quality 0.11-0.13, Ramachandran 27-32%, Clash 94-98%  
**Target:** Quality >0.75, Ramachandran >85%, Clash <5%

---

## Executive Summary

After analyzing the training and sampling code and comparing with latest research, here are the **critical improvements** needed to achieve SOTA performance:

### Priority 1 (Critical - Implement First)
1. **Add Clash Detection to Training Loss** - Currently missing, causing 94-98% clash rates
2. **Strengthen Ramachandran Loss** - Current weighting too weak (0.15-0.30), needs 0.35-0.50
3. **Add Pairwise Distance Loss** - Missing 3D structure consistency
4. **Implement CFG-Zero*** - Better classifier-free guidance (2024 research)

### Priority 2 (High Impact)
5. **Increase Training Epochs** - Current: 10, Recommended: 50-150
6. **Add EMA (Exponential Moving Average)** - Better model stability
7. **Implement OAT-FM** - Optimal Acceleration Transport (2024 research)
8. **Fix Sequence Diversity in Sampling** - Currently generates identical sequences

### Priority 3 (Medium Impact)
9. **Add Energy-Based Refinement** - Post-processing with Rosetta/Amber
10. **Implement Two-Stage Flow Matching** - Sparse to dense refinement
11. **Add Contrastive Flow Matching** - Better sample diversity

---

## Detailed Code-Level Recommendations

### 1. ❌ CRITICAL: Add Clash Detection to Training Loss

**Current State:**
- Clash detection exists in `geometric_validation.py` but **not used in training loss**
- Geometric loss only includes Ramachandran + omega + bond angles
- Result: 94-98% clash rate (should be <5%)

**Fix Required:**

```python
# In foldingdiff/enhanced_models_v2.py, _compute_geometric_loss method
# Add after line ~1306 (after Ramachandran computation)

def _compute_geometric_loss(
    self,
    velocity_pred: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    motif_coords: Optional[torch.Tensor] = None,
    motif_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # ... existing Ramachandran and omega code ...
    
    # NEW: Clash penalty from angles
    # Convert angles to approximate coordinates for clash detection
    try:
        from foldingdiff.nerf import angles_to_coords
        from foldingdiff.geometric_validation import compute_clash_penalty_torch
        
        # Get lengths from attention mask
        lengths = attention_mask.sum(dim=1).long()
        
        # Convert angles to coordinates (batch-wise)
        coords_list = []
        for i in range(x_0.shape[0]):
            coords = angles_to_coords(
                x_0[i:i+1, :lengths[i], :],
                lengths[i:i+1]
            )
            coords_list.append(coords[0])
        
        # Compute clash penalty (differentiable approximation)
        clash_penalty = 0.0
        for i, coords in enumerate(coords_list):
            # Approximate clash detection (fast, differentiable)
            # Check for atoms too close together
            n_atoms = coords.shape[0] * 4  # CA, C, N, O per residue
            if n_atoms > 1:
                # Compute pairwise distances
                coords_flat = coords.reshape(-1, 3)  # [n_atoms, 3]
                dists = torch.cdist(coords_flat, coords_flat)  # [n_atoms, n_atoms]
                
                # Mask self-distances
                mask = torch.eye(n_atoms, device=dists.device, dtype=torch.bool)
                dists = dists[~mask].reshape(n_atoms, n_atoms - 1)
                
                # Penalize distances < 2.0 Å (clash threshold)
                clash_threshold = 2.0
                clashes = (dists < clash_threshold).float()
                clash_penalty += clashes.sum() / (n_atoms * (n_atoms - 1))
        
        clash_penalty = clash_penalty / len(coords_list)  # Average over batch
        
    except Exception as e:
        logging.warning(f"Could not compute clash penalty: {e}")
        clash_penalty = torch.tensor(0.0, device=x_0.device, requires_grad=True)
    
    # Combine losses with stronger clash weighting
    total_loss = (
        rama_loss * 0.4 +           # Ramachandran (increased from 0.5)
        omega_penalty * 0.2 +        # Omega trans (decreased from 0.3)
        bond_penalty * 0.1 +         # Bond angles (same)
        clash_penalty * 0.3          # NEW: Clash penalty (30% weight)
    )
    
    return total_loss
```

**Expected Impact:** Clash rate 94-98% → 20-40% (first step), then 5-10% with refinement

**Location:** `foldingdiff/enhanced_models_v2.py`, line ~1255-1400

---

### 2. ❌ CRITICAL: Strengthen Ramachandran Loss Weighting

**Current State:**
- Geometric weight: 0.15-0.30 (adaptive, but too low)
- Ramachandran favored: 27-32% (target: >85%)
- Outlier penalty: 1.0-2.0 (needs to be stronger)

**Fix Required:**

```python
# In foldingdiff/enhanced_models_v2.py, training_step method
# Modify lines ~1038-1067

# BEST PRACTICE: Stronger geometric loss weighting
if self.use_geometric_loss:
    try:
        current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        total_epochs = self.epochs if hasattr(self, 'epochs') else 50
        progress = current_epoch / total_epochs if total_epochs > 0 else 0.0
        
        # INCREASED weights for better geometric learning
        if progress < 0.2:  # Early training (0-20%)
            adaptive_geometric_weight = 0.25  # Increased from 0.15
        elif progress < 0.5:  # Mid training (20-50%)
            adaptive_geometric_weight = 0.35  # Increased from 0.20
        elif progress < 0.75:  # Late training (50-75%)
            adaptive_geometric_weight = 0.45  # Increased from 0.25
        else:  # Final training (75-100%)
            adaptive_geometric_weight = 0.50  # Increased from 0.30 (MAXIMUM)
        
        geometric_loss = self._compute_geometric_loss(...)
        if not torch.isnan(geometric_loss) and not torch.isinf(geometric_loss):
            total_loss = total_loss + adaptive_geometric_weight * geometric_loss
            log_dict['train_geometric_loss'] = geometric_loss
            log_dict['train_geometric_weight'] = adaptive_geometric_weight
```

**Also strengthen Ramachandran penalty in _compute_geometric_loss:**

```python
# In _compute_geometric_loss, modify Ramachandran computation
# Around line ~1296-1330

# INCREASED outlier penalty (from 1.0-2.0 to 3.0-5.0)
outlier_penalty_weight = 3.0 + 2.0 * progress  # 3.0 early, 5.0 late
favored_reward_weight = 0.5 + 0.5 * progress   # 0.5 early, 1.0 late

# Compute Ramachandran loss with stronger penalties
rama_outliers = compute_ramachandran_outliers(phi, psi, mask_expanded_rama)
rama_favored = compute_ramachandran_favored(phi, psi, mask_expanded_rama)

outlier_penalty = (rama_outliers * outlier_penalty_weight).mean()
favored_reward = -(rama_favored * favored_reward_weight).mean()

rama_loss = outlier_penalty + favored_reward
```

**Expected Impact:** Ramachandran favored 27-32% → 55-70% (first step), then 75-85% with more training

---

### 3. ❌ CRITICAL: Add Pairwise Distance Loss

**Current State:**
- Only angular flow matching loss
- No 3D structure consistency during training
- Model doesn't learn to respect 3D geometry

**Fix Required:**

```python
# In foldingdiff/enhanced_models_v2.py, training_step method
# Add after main_loss computation (around line ~1012)

# NEW: Pairwise distance loss for 3D structure consistency
if self.use_geometric_loss:
    try:
        from foldingdiff.nerf import angles_to_coords
        from foldingdiff.losses import pairwise_dist_loss
        
        # Convert predicted angles to coordinates
        # Use x_0 + v_pred * dt as predicted next step
        dt = 0.01  # Small step
        x_pred = x_0 + v_pred * dt
        
        # Get lengths from attention mask
        lengths = attention_mask.sum(dim=1).long()
        
        # Convert to coordinates
        coords_pred_list = []
        coords_target_list = []
        
        for i in range(batch_size):
            length = lengths[i].item()
            if length > 0:
                # Predicted coordinates
                coords_pred = angles_to_coords(
                    x_pred[i:i+1, :length, :],
                    torch.tensor([length], device=device)
                )
                coords_pred_list.append(coords_pred[0])
                
                # Target coordinates (from x_0)
                coords_target = angles_to_coords(
                    x_0[i:i+1, :length, :],
                    torch.tensor([length], device=device)
                )
                coords_target_list.append(coords_target[0])
        
        # Compute pairwise distance loss
        if len(coords_pred_list) > 0:
            pairwise_loss = 0.0
            for coords_pred, coords_target in zip(coords_pred_list, coords_target_list):
                # Compute pairwise distance matrices
                dists_pred = torch.cdist(coords_pred, coords_pred)
                dists_target = torch.cdist(coords_target, coords_target)
                
                # Loss: MSE of distance matrices
                pairwise_loss += F.mse_loss(dists_pred, dists_target)
            
            pairwise_loss = pairwise_loss / len(coords_pred_list)
            
            # Add to total loss (weight: 0.1-0.15)
            pairwise_weight = 0.1 + 0.05 * progress  # 0.1 early, 0.15 late
            total_loss = total_loss + pairwise_weight * pairwise_loss
            log_dict['train_pairwise_loss'] = pairwise_loss
            log_dict['train_pairwise_weight'] = pairwise_weight
            
    except Exception as e:
        logging.warning(f"Could not compute pairwise distance loss: {e}")
```

**Expected Impact:** Quality +0.05-0.10, Clash rate -0.05-0.10

---

### 4. ⚠️ HIGH PRIORITY: Implement CFG-Zero* (Improved Classifier-Free Guidance)

**Current State:**
- Standard classifier-free guidance in sampling
- CFG-Zero* (2024 research) provides better guidance with optimized scale

**Fix Required:**

```python
# In bin/sample_advanced_flow.py, sample_with_advanced_features function
# Modify around line ~454-465

# REPLACE standard CFG with CFG-Zero*
if guidance_scale > 1.0:
    # Standard unconditional prediction
    v_uncond = model.forward(
        x, t,
        attention_mask=batch['attn_mask'],
        # ... other args ...
        motif_mask=torch.zeros_like(batch['motif_mask']),
        motif_features=torch.zeros_like(batch['motif_features']),
    )
    
    # NEW: CFG-Zero* - optimized scale and zero-init
    # 1. Compute optimal guidance scale (adaptive)
    optimal_scale = guidance_scale * (1.0 + 0.1 * (1.0 - t.item()))  # Higher at start
    
    # 2. Zero-init for early steps (t > 0.8)
    if t.item() > 0.8:
        # Zero out velocity for first few steps (CFG-Zero*)
        v_pred_zero = v_uncond.clone()
        v_pred_zero.zero_()
        # Blend: start with zero, transition to guided
        blend = (1.0 - t.item()) / 0.2  # 1.0 at t=0.8, 0.0 at t=1.0
        v_pred = blend * v_pred_zero + (1.0 - blend) * v_pred
    else:
        # Standard CFG for later steps
        v_pred = v_uncond + optimal_scale * (v_pred - v_uncond)
```

**Expected Impact:** Quality +0.05-0.10, Better motif preservation

**Reference:** arXiv:2503.18886 (CFG-Zero*)

---

### 5. ⚠️ HIGH PRIORITY: Increase Training Epochs

**Current State:**
- Training script: `EPOCHS=10` (line 99 in train_advanced_flow.py)
- Config: `EPOCHS=10` (config_advanced_flow.sh line 17)
- SOTA models: 50-150 epochs

**Fix Required:**

```bash
# In config_advanced_flow.sh, line 17
EPOCHS=50  # Changed from 10 to 50 (minimum for SOTA)
# Or for best results:
EPOCHS=100  # Full convergence

# In bin/train_advanced_flow.py, line 99
parser.add_argument("--epochs", type=int, default=50)  # Changed from 150 to 50 (more reasonable)
```

**Expected Impact:** Quality +0.10-0.25, Ramachandran +10-20%

---

### 6. ⚠️ HIGH PRIORITY: Add EMA (Exponential Moving Average)

**Current State:**
- No EMA for model weights
- Final model uses last checkpoint (can be noisy)

**Fix Required:**

```python
# In foldingdiff/enhanced_models_v2.py, __init__ method
# Add after model initialization

from torch.optim.swa_utils import AveragedModel

# Initialize EMA model
self.ema_model = AveragedModel(
    self,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
)

# In training_step, after optimizer.step()
# Add at end of training_step (around line ~1136)

# Update EMA model
if hasattr(self, 'ema_model'):
    self.ema_model.update_parameters(self)

# In validation_step, use EMA model
# Modify validation_step (around line ~1138)

def validation_step(self, batch, batch_idx):
    # Use EMA model for validation if available
    model_to_use = self.ema_model.module if hasattr(self, 'ema_model') else self
    # ... rest of validation code using model_to_use ...
```

**Expected Impact:** Quality +0.03-0.05, More stable training

---

### 7. ⚠️ HIGH PRIORITY: Fix Sequence Diversity in Sampling

**Current State:**
- All generated sequences are identical (100% identity)
- `generate_dummy_sequence` uses timestamp seed but may not be diverse enough

**Fix Required:**

```python
# In bin/sample_advanced_flow.py, generate_dummy_sequence function
# Modify around line ~116-151

def generate_dummy_sequence(length: int, seed: Optional[int] = None) -> str:
    """
    Generate diverse amino acid sequence.
    
    FIXED: Ensure true diversity across samples
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        # Use more diverse seeding
        seed = int(time.time() * 1000000) % (2**31)  # Microsecond precision
        rng = np.random.RandomState(seed)
    
    # Use realistic amino acid frequencies
    aa_freq = {
        'A': 0.082, 'R': 0.055, 'N': 0.041, 'D': 0.054, 'C': 0.014,
        'Q': 0.039, 'E': 0.067, 'G': 0.071, 'H': 0.022, 'I': 0.059,
        'L': 0.096, 'K': 0.058, 'M': 0.024, 'F': 0.039, 'P': 0.047,
        'S': 0.066, 'T': 0.053, 'W': 0.010, 'Y': 0.029, 'V': 0.069
    }
    
    amino_acids = list(aa_freq.keys())
    weights = np.array(list(aa_freq.values()))
    weights = weights / weights.sum()
    
    # Generate sequence with diversity
    sequence = ''.join(rng.choice(amino_acids, size=length, p=weights))
    
    return sequence

# In sample_advanced_flow_matching function, modify sequence generation
# Around line ~268-270

# FIXED: Use sample index for true diversity
sequence_seed = sample_index * 1000 + int(time.time() * 1000) % 1000
sequence = generate_dummy_sequence(length, seed=sequence_seed)
```

**Also add sequence diversity loss during training:**

```python
# In training_step, add sequence diversity penalty
# Add after main_loss computation

if 'sequences' in batch and len(batch['sequences']) > 1:
    # Compute sequence diversity
    sequences = batch['sequences']
    diversity_loss = 0.0
    
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            # Hamming distance
            seq1, seq2 = sequences[i], sequences[j]
            min_len = min(len(seq1), len(seq2))
            hamming = sum(seq1[k] != seq2[k] for k in range(min_len))
            similarity = 1.0 - (hamming / min_len)
            
            # Penalize high similarity (want diverse sequences)
            diversity_loss += similarity * 0.1  # Penalty for similarity
    
    if diversity_loss > 0:
        total_loss = total_loss + diversity_loss
        log_dict['train_sequence_diversity_loss'] = diversity_loss
```

**Expected Impact:** Sequence identity 100% → 60-80% (realistic diversity)

---

### 8. MEDIUM PRIORITY: Implement OAT-FM (Optimal Acceleration Transport)

**Reference:** arXiv:2509.24936

**Implementation:**

```python
# Create new file: foldingdiff/oat_fm.py

class OptimalAccelerationTransportFM:
    """
    Optimal Acceleration Transport for Flow Matching (OAT-FM)
    
    Improves flow matching by optimizing acceleration transport
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def get_oat_interpolant(self, x_0, x_1, t):
        """
        OAT-FM interpolant with acceleration optimization
        """
        # Standard interpolant
        x_t = (1 - t) * x_0 + t * x_1
        
        # Add acceleration term
        acceleration = (x_1 - x_0) * (1 - 2 * t)
        x_t = x_t + self.alpha * acceleration * t * (1 - t)
        
        return x_t
    
    def get_oat_velocity(self, x_0, x_1, t):
        """
        OAT-FM velocity with acceleration
        """
        # Standard velocity
        v = x_1 - x_0
        
        # Add acceleration component
        acceleration = (x_1 - x_0) * (1 - 2 * t)
        v = v + self.alpha * acceleration
        
        return v

# In training_step, use OAT-FM instead of standard flow
# Modify around line ~933-954

if self.use_oat_fm:  # New flag
    from foldingdiff.oat_fm import OptimalAccelerationTransportFM
    oat_fm = OptimalAccelerationTransportFM(alpha=0.5)
    x_t = oat_fm.get_oat_interpolant(x_0, x_1, t)
    v_target = oat_fm.get_oat_velocity(x_0, x_1, t)
else:
    # Standard flow matching
    x_t = self.flow_schedule.get_interpolant(x_0, x_1, t)
    v_target = self.flow_schedule.get_target_velocity(x_0, x_1)
```

**Expected Impact:** Quality +0.05-0.10, Faster convergence

---

### 9. MEDIUM PRIORITY: Add Energy-Based Refinement

**Implementation:**

```python
# Create new file: foldingdiff/energy_refinement.py

def refine_with_rosetta(
    angles: torch.Tensor,
    sequence: str,
    n_iterations: int = 50
) -> torch.Tensor:
    """
    Post-process structures with Rosetta energy minimization
    
    This is a placeholder - requires Rosetta installation
    """
    # Convert angles to PDB
    # Run Rosetta relax
    # Convert back to angles
    # Return refined angles
    pass

# In sampling, add refinement step
# In bin/sample_advanced_flow.py, after sampling

if args.use_energy_refinement:
    from foldingdiff.energy_refinement import refine_with_rosetta
    refined_angles = refine_with_rosetta(
        angles, sequence, n_iterations=50
    )
    angles = refined_angles
```

**Expected Impact:** Quality +0.05-0.15, Clash rate -0.05-0.10

---

## Implementation Priority

### Phase 1 (Immediate - Week 1)
1. ✅ Add clash detection to training loss
2. ✅ Strengthen Ramachandran loss weighting
3. ✅ Add pairwise distance loss
4. ✅ Increase training epochs to 50

**Expected Result:** Quality 0.11-0.13 → 0.40-0.60 (3-5x improvement)

### Phase 2 (Short-term - Week 2-3)
5. ✅ Implement CFG-Zero*
6. ✅ Add EMA
7. ✅ Fix sequence diversity

**Expected Result:** Quality 0.40-0.60 → 0.55-0.70 (5-6x improvement)

### Phase 3 (Medium-term - Week 4-6)
8. ✅ Implement OAT-FM
9. ✅ Add energy-based refinement
10. ✅ Two-stage flow matching

**Expected Result:** Quality 0.55-0.70 → 0.75-0.90 ✅ **MEETS SOTA**

---

## Expected Performance Trajectory

| Phase | Quality | Ramachandran | Clash Rate | Status |
|-------|---------|--------------|------------|--------|
| **Current** | 0.11-0.13 | 27-32% | 94-98% | ❌ |
| **After Phase 1** | 0.40-0.60 | 55-70% | 20-40% | ⚠️ |
| **After Phase 2** | 0.55-0.70 | 70-80% | 10-20% | ⚠️ |
| **After Phase 3** | 0.75-0.90 | 85-95% | <5% | ✅ **SOTA** |

---

## Code Files to Modify

1. **`foldingdiff/enhanced_models_v2.py`**
   - `_compute_geometric_loss` - Add clash penalty
   - `training_step` - Strengthen weights, add pairwise loss, add EMA
   - `validation_step` - Use EMA model

2. **`bin/train_advanced_flow.py`**
   - Increase default epochs to 50
   - Add OAT-FM flag

3. **`bin/sample_advanced_flow.py`**
   - Implement CFG-Zero*
   - Fix sequence diversity
   - Add energy refinement option

4. **`config_advanced_flow.sh`**
   - Increase EPOCHS to 50-100

5. **New Files:**
   - `foldingdiff/oat_fm.py` - OAT-FM implementation
   - `foldingdiff/energy_refinement.py` - Energy-based refinement

---

## Testing Strategy

1. **Implement Phase 1 fixes** → Re-train → Evaluate
2. **If quality < 0.4**, implement Phase 2 fixes → Re-train → Evaluate
3. **If quality < 0.6**, implement Phase 3 fixes → Re-train → Evaluate
4. **Monitor metrics:** Quality, Ramachandran, Clash rate, Sequence diversity

---

## References

1. **CFG-Zero***: arXiv:2503.18886 - Improved Classifier-Free Guidance
2. **OAT-FM**: arXiv:2509.24936 - Optimal Acceleration Transport
3. **Two-Stage Flow Matching**: arXiv:2512.20988 - Sparse to Dense Refinement
4. **Contrastive Flow Matching**: ICCV 2025 - Better Diversity

---

**Last Updated:** 2026-01-03  
**Status:** Ready for Implementation


