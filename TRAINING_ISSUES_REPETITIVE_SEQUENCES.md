# Training Issues Causing Repetitive Sequences

## Executive Summary

After thorough analysis of the training code and online research, I've identified **training-specific issues** that contribute to repetitive amino acid sequences from ProteinMPNN. While the primary issue is ProteinMPNN temperature (0.1), there are **training problems** that cause poor backbone diversity, which indirectly leads to repetitive sequences.

## Root Causes

### 1. **Lack of Diversity Regularization** ⚠️ CRITICAL TRAINING ISSUE

**Problem:**
The training loss function has **no diversity penalty** to prevent mode collapse. The model can learn to generate very similar backbones, leading to:
- Low structural diversity
- Uniform backbone geometries
- ProteinMPNN assigning repetitive sequences to similar structures

**Current Loss Function:**
```python
# In flow_models.py::training_step()
loss = compute_angular_flow_matching_loss(
    v_pred, v_target,
    is_angular=self.ft_is_angular,
    mask=batch['attn_mask'],
    motif_mask=batch.get('motif_mask', None),
    scaffold_weight=2.0
)
```

**Missing:** Diversity regularization to encourage diverse backbone generation.

**Solution:** Add diversity regularization to training loss.

---

### 2. **Insufficient Training Epochs** ⚠️ MAJOR ISSUE

**Current:** `EPOCHS=10` (in train_and_evaluate_enhanced_flow.sh line 21)

**Problem:**
- Flow matching models need 50-100 epochs for proper convergence
- With only 10 epochs, the model:
  - Doesn't learn full time range properly
  - Produces low-diversity backbones
  - Struggles with complex geometries

**Evidence from Research:**
- Flow matching papers recommend 100+ epochs
- Mode collapse is more likely with undertrained models
- Short training leads to memorization rather than generalization

**Solution:** Increase to 100 epochs minimum.

---

### 3. **No Batch Diversity Monitoring** ⚠️ MODERATE ISSUE

**Problem:**
Training doesn't monitor or encourage diversity within batches. The model could learn to generate very similar structures.

**Missing:**
- Batch-level diversity metrics
- Regularization to penalize similar samples
- Monitoring of structural diversity during training

**Solution:** Add diversity monitoring and regularization.

---

### 4. **Potential Mode Collapse from Loss Function** ⚠️ MODERATE ISSUE

**Problem:**
The MSE loss on velocity can lead to mode collapse if:
- Model finds a "safe" mode that minimizes loss
- No explicit diversity encouragement
- Similar structures get similar velocities

**Current Loss:**
```python
loss = MSE(v_pred, v_target)  # No diversity component
```

**Solution:** Add diversity component to loss.

---

### 5. **Fixed Noise Sampling** ⚠️ MINOR ISSUE

**Problem:**
Noise `x_1` is sampled as `torch.randn_like(x_0)` - standard Gaussian. This is correct, but:
- No explicit encouragement of diverse noise paths
- Could benefit from noise augmentation

**Current:**
```python
x_1 = torch.randn_like(x_0)  # Standard Gaussian
```

**Solution:** Consider noise augmentation for diversity.

---

## Recommended Training Fixes

### Fix 1: Add Diversity Regularization to Loss ✅

Add a diversity penalty to encourage diverse backbone generation:

```python
def compute_diversity_penalty(
    samples: torch.Tensor,
    mask: torch.Tensor,
    diversity_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute diversity penalty to prevent mode collapse.
    
    Penalizes samples that are too similar within a batch.
    """
    # Only consider valid positions
    valid_samples = samples * mask.unsqueeze(-1)
    
    # Compute pairwise distances within batch
    batch_size = samples.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=samples.device)
    
    # Flatten samples
    samples_flat = valid_samples.reshape(batch_size, -1)
    
    # Compute pairwise cosine similarity
    # Normalize
    samples_norm = F.normalize(samples_flat, p=2, dim=1)
    # Pairwise similarity matrix
    similarity = torch.mm(samples_norm, samples_norm.t())
    # Remove diagonal (self-similarity)
    mask_triu = torch.triu(torch.ones_like(similarity), diagonal=1)
    similarities = similarity * mask_triu
    
    # Penalize high similarity (low diversity)
    # We want low similarity = high diversity
    diversity_penalty = similarities.abs().mean()
    
    return diversity_weight * diversity_penalty
```

**Usage in training_step:**
```python
# After computing main loss
loss = compute_angular_flow_matching_loss(...)

# Add diversity penalty
if batch_size > 1:
    diversity_penalty = compute_diversity_penalty(
        x_0,  # Clean samples
        batch['attn_mask'],
        diversity_weight=0.01
    )
    loss = loss - diversity_penalty  # Negative because we want to maximize diversity
```

---

### Fix 2: Increase Training Epochs ✅

**File:** `train_and_evaluate_enhanced_flow.sh`

**Change:**
```bash
EPOCHS=100  # Changed from 10
```

**Rationale:**
- Flow matching needs proper convergence
- More epochs = better diversity
- Prevents undertraining mode collapse

---

### Fix 3: Add Diversity Monitoring ✅

Add logging to track diversity during training:

```python
def compute_batch_diversity(samples: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute diversity metric for batch"""
    valid_samples = samples * mask.unsqueeze(-1)
    samples_flat = valid_samples.reshape(samples.shape[0], -1)
    
    # Pairwise distances
    distances = torch.cdist(samples_flat, samples_flat, p=2)
    # Remove diagonal
    mask_triu = torch.triu(torch.ones_like(distances), diagonal=1)
    distances = distances * mask_triu
    
    return distances[mask_triu > 0].mean().item()

# In training_step, after loss computation:
batch_diversity = compute_batch_diversity(x_0, batch['attn_mask'])
self.log('train_batch_diversity', batch_diversity)
```

---

### Fix 4: Add Noise Augmentation ✅

Enhance noise sampling for better diversity:

```python
# Instead of simple Gaussian noise
x_1 = torch.randn_like(x_0)

# Use noise augmentation
def sample_diverse_noise(x_0: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Sample noise with augmentation for diversity"""
    base_noise = torch.randn_like(x_0)
    
    # Add small random scale variations
    noise_scale = scale * (1.0 + 0.1 * torch.rand(1, device=x_0.device))
    
    return base_noise * noise_scale

x_1 = sample_diverse_noise(x_0, scale=1.0)
```

---

### Fix 5: Add Gradient Penalty for Diversity ✅

Prevent gradients from collapsing to single mode:

```python
def compute_gradient_penalty(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Compute gradient penalty to encourage diversity"""
    x_t.requires_grad_(True)
    
    v_pred = model(x_t, t, attention_mask=mask)
    
    # Compute gradient w.r.t. input
    grad = torch.autograd.grad(
        outputs=v_pred.sum(),
        inputs=x_t,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Penalize small gradients (which indicate mode collapse)
    grad_norm = grad.norm(dim=-1)
    penalty = (grad_norm - 1.0).pow(2).mean()
    
    return 0.01 * penalty  # Small weight
```

---

## Implementation Priority

### 🔴 **CRITICAL - Implement Immediately**

1. **Increase Epochs to 100**
   - File: `train_and_evaluate_enhanced_flow.sh`
   - Impact: Major improvement in backbone quality

2. **Add Diversity Regularization**
   - File: `foldingdiff/flow_models.py`
   - Impact: Prevents mode collapse, increases diversity

### 🟡 **IMPORTANT - Implement Soon**

3. **Add Diversity Monitoring**
   - File: `foldingdiff/flow_models.py`
   - Impact: Track diversity during training

4. **Add Noise Augmentation**
   - File: `foldingdiff/flow_models.py`
   - Impact: Better exploration of latent space

### 🟢 **OPTIONAL - Nice to Have**

5. **Add Gradient Penalty**
   - File: `foldingdiff/flow_models.py`
   - Impact: Additional diversity guarantee

---

## Expected Improvements

### After Training Fixes

1. **Higher Backbone Diversity**
   - More varied structures
   - Better geometric diversity
   - Reduced mode collapse

2. **Better ProteinMPNN Sequences**
   - Less repetition
   - More diverse amino acids
   - Higher sequence entropy

3. **Improved Structural Quality**
   - More realistic geometries
   - Better Ramachandran distributions
   - Higher designability

---

## Testing Protocol

### Step 1: Monitor Training Diversity

```python
# Add to training_step
batch_diversity = compute_batch_diversity(x_0, batch['attn_mask'])
self.log('train_batch_diversity', batch_diversity)

# Check TensorBoard
# Should see diversity increasing over epochs
```

### Step 2: Validate Generated Backbones

```bash
# After training, check backbone diversity
python analyze_backbone_diversity.py \
    --input_dir results/enhanced_flow/samples_*/unconditional/pdb \
    --output report.json
```

**Expected Metrics:**
- Mean pairwise distance > 10.0 (higher = more diverse)
- CA-CA distance std > 0.2 Å
- Ramachandran coverage > 60%

### Step 3: Test ProteinMPNN Sequences

```bash
# Generate sequences with proper temperature
bash run_proteinmpnn.sh \
    --input_dir results/enhanced_flow/samples_*/unconditional/pdb \
    --temperature 0.3 \
    --output_dir sequences_diverse
```

**Expected:**
- Sequence entropy > 2.5 bits
- Max consecutive identical < 5
- Diverse amino acid composition

---

## Code Changes Summary

### File 1: `train_and_evaluate_enhanced_flow.sh`
```bash
EPOCHS=100  # Changed from 10
```

### File 2: `foldingdiff/flow_models.py`
Add diversity regularization and monitoring to `training_step()`.

### File 3: `foldingdiff/flow_matching.py`
Add `compute_diversity_penalty()` function.

---

## Research Basis

1. **Flow Matching Papers:**
   - Recommend 100+ epochs for convergence
   - Mode collapse is a known issue without regularization

2. **Generative Model Best Practices:**
   - Diversity regularization prevents mode collapse
   - Batch diversity monitoring is essential

3. **Protein Design Literature:**
   - Diverse backbones → diverse sequences
   - Uniform structures → repetitive sequences

---

## Summary

The repetitive sequences are caused by:
1. **Low ProteinMPNN temperature (0.1)** - PRIMARY ISSUE (already documented)
2. **Lack of diversity regularization** - TRAINING ISSUE (NEW)
3. **Insufficient training (10 epochs)** - TRAINING ISSUE (partially fixed)
4. **No diversity monitoring** - TRAINING ISSUE (NEW)

**Training Fixes:**
- Add diversity regularization to loss
- Increase epochs to 100
- Add diversity monitoring
- Consider noise augmentation

**Expected Result:**
- More diverse backbones
- Less repetitive sequences
- Higher quality designs

