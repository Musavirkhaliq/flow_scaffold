# Model Improvement Plan
## Systematic Testing and Optimization

**Date:** 2026-01-03  
**Status:** Ready to test

---

## 🎯 Goal

Find optimal loss weight configuration that maximizes:
1. **Trans Fraction** (>95%) - CRITICAL
2. **Quality Score** (>0.3)
3. **Ramachandran Favored** (>50%)
4. **Clash Rate** (<0.2)

---

## 📊 Current Baseline

From fast test run:
- Quality: 0.142
- Ramachandran: 35.4%
- Trans: 78-85% (should be >95%)
- Clash: 0.725

**Current Loss Weights:**
- Omega: 1.0
- Ramachandran: 2.0
- Geometric: 0.25

---

## 🧪 Test Configurations

### Configuration 1: Baseline (Current)
- Omega: 1.0
- Ramachandran: 2.0
- Geometric: 0.25
- **Expected:** Strong constraints, may cause instability

### Configuration 2: Moderate
- Omega: 0.5
- Ramachandran: 1.5
- Geometric: 0.20
- **Expected:** Balanced, more stable training

### Configuration 3: Weak
- Omega: 0.3
- Ramachandran: 1.0
- Geometric: 0.15
- **Expected:** Less aggressive, may not learn constraints well

### Configuration 4: Strong Omega Only
- Omega: 1.0
- Ramachandran: 1.0
- Geometric: 0.25
- **Expected:** Focus on omega fix

### Configuration 5: Strong Rama Only
- Omega: 0.3
- Ramachandran: 2.0
- Geometric: 0.25
- **Expected:** Focus on Ramachandran quality

### Configuration 6: Balanced
- Omega: 0.7
- Ramachandran: 1.5
- Geometric: 0.20
- **Expected:** Moderate on both, balanced

---

## 🚀 Execution Plan

### Step 1: Run Systematic Tests

```bash
bash test_loss_configurations.sh
```

This will:
- Test all 6 configurations
- Each takes ~10-15 minutes
- Total time: ~1-1.5 hours
- Generate comparison table

### Step 2: Analyze Results

Check the comparison table for:
1. **Highest trans fraction** (target: >95%)
2. **Highest quality score** (target: >0.3)
3. **Highest Ramachandran favored** (target: >50%)
4. **Lowest clash rate** (target: <0.2)
5. **Best composite score**

### Step 3: Apply Best Configuration

```bash
# Apply best weights
python apply_best_loss_weights.py \
    --omega_weight <best_omega> \
    --rama_penalty <best_rama>

# Update geometric weight in training script
# Edit train_and_evaluate_advanced_flow.sh: GEOMETRIC_WEIGHT=<best_geo>
```

### Step 4: Run Full Training

```bash
bash train_and_evaluate_advanced_flow.sh
```

---

## 📈 Expected Improvements

### After Finding Optimal Weights:
- Trans Fraction: 78-85% → **>95%** ✅
- Quality: 0.142 → **0.25-0.40** (2-3x improvement)
- Ramachandran: 35.4% → **45-60%** (1.3-1.7x improvement)
- Clash Rate: 0.725 → **0.3-0.5** (2x improvement)

### After Full Training (20+ epochs):
- Quality: **0.40-0.60** (3-4x improvement)
- Ramachandran: **55-75%** (1.5-2x improvement)
- Clash Rate: **0.15-0.30** (2-5x improvement)
- Trans Fraction: **>95%** ✅

---

## 🔍 What to Look For

### Good Signs:
- ✅ Trans fraction increasing (toward >95%)
- ✅ Quality score increasing (toward >0.3)
- ✅ Ramachandran favored increasing (toward >50%)
- ✅ Clash rate decreasing (toward <0.2)
- ✅ Training loss decreasing smoothly

### Bad Signs:
- ❌ Trans fraction stuck or decreasing
- ❌ Quality score decreasing
- ❌ Training loss unstable or increasing
- ❌ NaN/Inf errors

---

## 🎯 Success Criteria

**Best configuration should have:**
1. Trans fraction >90% (ideally >95%)
2. Quality score >0.20 (ideally >0.30)
3. Ramachandran favored >40% (ideally >50%)
4. Clash rate <0.6 (ideally <0.3)
5. Stable training (no NaN/Inf)

---

## 📝 Notes

- Fast tests use 3 epochs + toy dataset (quick but approximate)
- Full training uses 20+ epochs + full dataset (slow but accurate)
- Results may vary between fast and full training
- Focus on trans fraction first (most critical issue)

---

**Ready to start testing!**



