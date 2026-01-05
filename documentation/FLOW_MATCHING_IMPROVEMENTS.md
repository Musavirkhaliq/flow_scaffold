# Flow Matching Improvements for Motif Scaffolding

## Summary

This document outlines the improvements made to the flow matching training code for motif scaffolding, based on best practices from recent literature and analysis of the codebase.

## Key Improvements

### 1. **Importance-Weighted Time Sampling** ✅

**Problem**: Uniform time sampling doesn't emphasize critical timesteps (t=0 and t=1) where the flow is most important.

**Solution**: Implemented Beta distribution sampling that concentrates samples near boundaries:
- Uses `Beta(α, α)` distribution with α=2.0
- More samples near t=0 (data) and t=1 (noise)
- Improves learning of critical flow transitions

**Location**: `flow_matching.py::FlowMatchingSchedule.sample_time()`

**Usage**:
```python
t = schedule.sample_time(batch_size, device, importance_weighting=True, alpha=2.0)
```

### 2. **Curriculum Learning for Time Sampling** ✅

**Problem**: Early training struggles with difficult timesteps.

**Solution**: Progressive curriculum:
- **Early training** (0-20% epochs): Focus on t ∈ [0, 0.5] (easier)
- **Mid training** (20-50% epochs): Expand to t ∈ [0.1, 0.9]
- **Late training** (50%+ epochs): Full importance-weighted sampling

**Location**: `flow_models.py::BertForFlowMatchingTraining.training_step()`

### 3. **Scaffold-Region Loss Weighting** ✅

**Problem**: Model needs to focus more on learning scaffold generation than preserving motifs (which are fixed).

**Solution**: Weight scaffold regions 2x more heavily in loss:
- Motif regions: weight = 1.0
- Scaffold regions: weight = 2.0
- Encourages better scaffold generation quality

**Location**: `flow_matching.py::compute_angular_flow_matching_loss()`

**Usage**:
```python
loss = compute_angular_flow_matching_loss(
    v_pred, v_target,
    is_angular=is_angular,
    mask=attn_mask,
    motif_mask=motif_mask,
    scaffold_weight=2.0  # Weight scaffold regions more
)
```

### 4. **Smooth Motif-Scaffold Transitions** ✅

**Problem**: Hard boundaries between motif and scaffold regions create discontinuities that hurt training.

**Solution**: Gaussian smoothing of motif mask boundaries:
- Smooths transition regions (default: 10% of sequence length)
- Prevents sharp discontinuities
- Helps model learn better boundary conditions

**Location**: `flow_matching.py::ConditionalFlowMatching.get_conditional_interpolant()`

**Usage**:
```python
x_t = conditional_flow.get_conditional_interpolant(
    x_0, x_1, t, motif_mask,
    smooth_transition=True,
    transition_width=0.1
)
```

### 5. **Stochastic Centering for Motifs** ✅

**Problem**: Model can learn fixed offsets between scaffold and motif, reducing generalization.

**Solution**: Add small random noise to motif angles during training:
- Prevents learning fixed positional relationships
- Improves inference-time generalization
- Standard deviation: 0.05 (small, preserves motif geometry)

**Location**: `motif_scaffolding.py::MotifScaffoldingDataset.__getitem__()`

**Usage**:
```python
item = dataset[index]  # Automatically applies stochastic centering
# Or disable: dataset.__getitem__(index, stochastic_centering=False)
```

### 6. **Improved Classifier-Free Guidance** ✅

**Problem**: Guidance dropout wasn't fully removing conditioning.

**Solution**: Enhanced dropout strategy:
- Zeros out both `motif_mask` AND `motif_angles`
- Ensures true unconditional generation
- Enables better classifier-free guidance during sampling

**Location**: `flow_models.py::BertForFlowMatchingTraining._apply_guidance_dropout()`

## Implementation Details

### Files Modified

1. **`foldingdiff/flow_matching.py`**
   - Added importance-weighted time sampling
   - Added smooth transition option to conditional interpolant
   - Enhanced loss function with scaffold weighting

2. **`foldingdiff/flow_models.py`**
   - Added curriculum learning for time sampling
   - Updated loss computation to use scaffold weighting
   - Improved guidance dropout
   - Enabled smooth transitions in conditional flow

3. **`foldingdiff/enhanced_models.py`**
   - Applied all improvements to enhanced training class
   - Consistent with base flow matching improvements

4. **`foldingdiff/motif_scaffolding.py`**
   - Added stochastic centering option

## Expected Benefits

1. **Better Convergence**: Curriculum learning and importance weighting improve early training stability
2. **Higher Quality Scaffolds**: Scaffold weighting focuses model on generation quality
3. **Smoother Boundaries**: Smooth transitions prevent training instabilities
4. **Better Generalization**: Stochastic centering prevents overfitting to fixed motif positions
5. **Stronger Guidance**: Improved classifier-free guidance enables better sampling control

## Research Basis

These improvements are based on:
- **Flow Matching** (Lipman et al., 2023): Importance weighting and optimal transport
- **FrameFlow** (Microsoft): Motif amortization and stochastic centering
- **RFdiffusion2**: Multi-motif scaffolding and flexible conditioning
- **Best Practices**: Time sampling strategies and loss weighting for conditional generation

## Configuration

All improvements are enabled by default. To adjust:

```python
# In training script
model = BertForFlowMatchingTraining(
    ...
    guidance_dropout=0.1,  # Adjust guidance strength
)

# In dataset creation
dataset = MotifScaffoldingDataset(
    ...
    # Stochastic centering is enabled by default
    # Can disable per-item: dataset.__getitem__(idx, stochastic_centering=False)
)
```

## Testing Recommendations

1. **Monitor training curves**: Should see faster convergence and lower loss
2. **Check scaffold quality**: Generated scaffolds should be more diverse and higher quality
3. **Validate motif preservation**: Motifs should still be preserved accurately
4. **Test guidance**: Classifier-free guidance should provide better control

## Future Improvements

Potential additional enhancements:
1. **Optimal Transport Flow Matching**: Better paths between noise and data
2. **Multi-Motif Support**: Handle multiple motifs with flexible positioning
3. **Adaptive Time Sampling**: Learn optimal time distribution during training
4. **Geometric Constraints**: Enforce physical constraints during flow

## References

- Lipman et al. (2023) "Flow Matching for Generative Modeling"
- FrameFlow: https://github.com/microsoft/protein-frame-flow
- RFdiffusion2: https://www.nature.com/articles/s41586-025-09746-w
- Proteina: https://github.com/NVIDIA-Digital-Bio/proteina

