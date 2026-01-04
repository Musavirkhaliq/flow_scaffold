# Training and Sampling Improvements (2025)

## Overview
Based on evaluation results showing poor model performance (97.8% clash rate, 64% Ramachandran outliers, 5% sequence recovery), comprehensive improvements have been implemented based on latest research (2024-2025) and best practices for flow matching models.

## Key Issues Identified
1. **High clash rate (97.8%)** - Geometric loss too strong
2. **High Ramachandran outliers (64%)** - Poor angle prediction
3. **Low sequence recovery (5%)** - Sequence-structure mismatch
4. **Low quality scores (0.067)** - Overall poor structure quality
5. **Sampling failures** - n_structures: 0 in all scenarios

## Training Improvements

### 1. Learning Rate Optimization
- **Reduced LR from 5e-5 to 3e-5** (config) and 1e-4 to 3e-5 (training script)
- **Increased warmup from 10% to 15%** for better stability
- **Rationale**: Lower learning rates are critical for flow matching stability (web research)

### 2. Gradient Clipping
- **Reduced from 1.0 to 0.5** in both config and training script
- **Rationale**: Prevents gradient explosion (was detecting gradients up to 35.8)

### 3. Geometric Loss Weight Reduction
- **Reduced from 0.2-0.5 to 0.10-0.18** (adaptive based on training progress)
- **Config default: 0.15** (reduced from 0.2)
- **Rationale**: High geometric loss was causing 97.8% clash rate

### 4. Time Sampling Improvements
- **Added U-shaped distribution** for rectified flows (web research: arxiv.org/abs/2405.20320)
- **Curriculum learning**: Early training focuses on easier timesteps (t near 0)
- **Adaptive alpha**: Increases from 1.5 to 3.0 as training progresses
- **Rationale**: Better time sampling improves flow matching performance

### 5. Loss Computation Improvements
- **Omega angle weighting**: Increased to 2.0x (was 1.5x) for trans preference
- **Huber loss for phi/psi**: Better gradient behavior for angular features
- **Adaptive scaffold weighting**: Based on motif complexity
- **Rationale**: Better handling of angular features improves structure quality

### 6. Epochs and Batch Size
- **Epochs: 30 → 50** (config updated)
- **Batch size: 16** (maintained with gradient accumulation)
- **Gradient accumulation: 4** (effective batch = 64)
- **Rationale**: More epochs needed for convergence (SOTA: 50-100 epochs)

## Sampling Improvements

### 1. Number of Steps
- **Increased from 50/150 to 200** (config and sampling script)
- **Rationale**: More steps = better quality (web research shows 200+ steps optimal)

### 2. ODE Solver
- **Changed default from "euler" to "rk4"** (Runge-Kutta 4th order)
- **Fixed RK4 implementation**: Proper 4-stage evaluation (was simplified)
- **Rationale**: RK4 is more accurate, reduces truncation errors

### 3. Guidance Scale
- **Maintained at 2.0** with adaptive scaling
- **CFG-Zero* improvements**: Optimized scale and zero-init for early steps
- **Rationale**: Better guidance improves conditional generation

### 4. Geometric Constraints
- **Better omega angle handling**: Enforce trans (π) after mean correction
- **Angle wrapping**: Proper modulo for angular features after each step
- **Rationale**: Prevents invalid angles that cause clashes

## Configuration Changes

### config_advanced_flow.sh
```bash
# Training
EPOCHS=50  # Increased from 30
LR=3e-5  # Reduced from 5e-5
WARMUP_RATIO=0.15  # Increased from 0.1
GRADIENT_CLIP=0.5  # Reduced from 1.0
GEOMETRIC_WEIGHT=0.15  # Reduced from 0.2

# Sampling
NUM_STEPS=200  # Increased from 150
SAMPLING_METHOD="rk4"  # Changed from "euler"
```

### Training Script (bin/train_advanced_flow.py)
- LR: 1e-4 → 3e-5
- Gradient clip: 1.0 → 0.5
- Geometric weight: 0.22 → 0.15

### Sampling Script (bin/sample_advanced_flow.py)
- Default steps: 50 → 200
- Default method: "euler" → "rk4"
- Fixed RK4 implementation

## Research-Based Improvements

### 1. U-Shaped Time Distribution
- **Source**: arxiv.org/abs/2405.20320 (Rectified Flows)
- **Implementation**: Mixture of Beta(0.5, 0.5) and Beta(alpha, alpha)
- **Benefit**: Better performance in low NFE settings

### 2. Minibatch Optimal Transport
- **Source**: Conditional Flow Matching (CFM) research
- **Status**: Framework exists, can be enabled with `--use_oat_fm`

### 3. Phase-Aware Training
- **Source**: arxiv.org/abs/2412.07972
- **Implementation**: Curriculum learning with adaptive time sampling
- **Benefit**: Better learning of probability and variance modes

### 4. Self-Corrected Flow Distillation
- **Source**: arxiv.org/abs/2412.16906
- **Status**: Can be added for one-step/few-step sampling

## Expected Improvements

1. **Clash Rate**: 97.8% → <20% (target: <10%)
2. **Ramachandran Outliers**: 64% → <20% (target: <10%)
3. **Sequence Recovery**: 5% → >30% (target: >50%)
4. **Quality Score**: 0.067 → >0.5 (target: >0.7)
5. **Sampling Success**: 0 structures → >80% success rate

## Next Steps

1. **Retrain model** with new parameters
2. **Evaluate** on test set
3. **Monitor** training metrics (loss, gradient norms, clash rates)
4. **Adjust** geometric weight if clash rate still high
5. **Consider** additional improvements:
   - Variational dequantization
   - Advanced coupling layers
   - Self-attention improvements
   - Ensemble methods

## Files Modified

1. `config_advanced_flow.sh` - Updated training and sampling parameters
2. `bin/train_advanced_flow.py` - Updated defaults and training logic
3. `bin/sample_advanced_flow.py` - Fixed RK4, increased steps, better guidance
4. `foldingdiff/enhanced_models_v2.py` - Reduced geometric loss weights
5. `foldingdiff/flow_matching.py` - Added U-shaped time distribution
6. `foldingdiff/flow_models.py` - Updated time sampling

## References

1. Rectified Flows: arxiv.org/abs/2405.20320
2. Phase-Aware Training: arxiv.org/abs/2412.07972
3. Self-Corrected Flow Distillation: arxiv.org/abs/2412.16906
4. Conditional Flow Matching: proceedings.iclr.cc/2025
5. Flow Matching Best Practices: emergentmind.com/topics/flow-matching-fm

