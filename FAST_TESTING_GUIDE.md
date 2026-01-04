# Fast Testing Guide
## Rapid Iteration for Loss Function Optimization

**Purpose:** Quickly test different loss function weights and configurations without waiting for full training.

---

## Quick Start

### Single Fast Test

```bash
# Test with default configuration
bash test_training_fast.sh
```

This will:
- Train for **3 epochs** (instead of 20)
- Use **toy dataset** (small subset)
- Generate **5 samples** (instead of 25)
- Use **50 sampling steps** (instead of 150)
- Complete in **~10-15 minutes** (instead of hours)

---

## Testing Different Loss Weights

### Option 1: Manual Testing

Edit `test_training_fast.sh` and change:

```bash
GEOMETRIC_WEIGHT=0.25  # Change this
# Or modify enhanced_models_v2.py directly for omega_weight and rama_penalty
```

Then run:
```bash
bash test_training_fast.sh
```

### Option 2: Python Script (More Control)

```bash
python test_training_fast.py \
    --output_dir results/test_fast/my_test \
    --experiment_name test_omega_0.5 \
    --geometric_weight 0.25 \
    --epochs 3 \
    --batch_size 32
```

---

## Recommended Test Configurations

### Test 1: Baseline (Current)
```bash
GEOMETRIC_WEIGHT=0.25
# Omega weight: 1.0 (in code)
# Rama penalty: 2.0 (in code)
```

### Test 2: Moderate Weights
```bash
GEOMETRIC_WEIGHT=0.15
# Omega weight: 0.5 (modify code)
# Rama penalty: 1.5 (modify code)
```

### Test 3: Weak Weights
```bash
GEOMETRIC_WEIGHT=0.10
# Omega weight: 0.3 (modify code)
# Rama penalty: 1.0 (modify code)
```

### Test 4: Strong Omega Only
```bash
GEOMETRIC_WEIGHT=0.25
# Omega weight: 1.0 (keep)
# Rama penalty: 1.0 (reduce)
```

### Test 5: Strong Rama Only
```bash
GEOMETRIC_WEIGHT=0.25
# Omega weight: 0.3 (reduce)
# Rama penalty: 2.0 (keep)
```

---

## Modifying Loss Weights in Code

To test different omega and Ramachandran weights, edit `foldingdiff/enhanced_models_v2.py`:

### Line ~1331: Ramachandran Loss
```python
# Current:
rama_loss = outlier_fraction * 2.0 - favored_fraction * 0.3

# Test with:
rama_loss = outlier_fraction * 1.5 - favored_fraction * 0.3  # Moderate
# or
rama_loss = outlier_fraction * 1.0 - favored_fraction * 0.5  # Weak
```

### Line ~1347: Omega Loss
```python
# Current:
omega_penalty = omega_penalty * 1.0  # Strong

# Test with:
omega_penalty = omega_penalty * 0.5  # Moderate
# or
omega_penalty = omega_penalty * 0.3  # Weak
```

---

## Comparing Results

After running multiple tests, compare:

```bash
# Check results directory
ls results/test_fast/

# Compare metrics
cat results/test_fast/*/evaluation/evaluation_summary.json | grep -A 5 "energy_plausibility"
```

Key metrics to compare:
1. **Trans Fraction** (target: >95%) - CRITICAL
2. **Quality Score** (target: >0.3)
3. **Ramachandran Favored** (target: >50%)
4. **Clash Rate** (target: <0.2)

---

## Quick Evaluation Script

Create a comparison table:

```bash
python << EOF
import json
import glob
from pathlib import Path

results = []
for path in glob.glob("results/test_fast/*/evaluation/evaluation_summary.json"):
    exp_name = Path(path).parent.parent.name
    with open(path) as f:
        data = json.load(f)
        ep = data.get('energy_plausibility', {})
        results.append({
            'name': exp_name,
            'quality': ep.get('mean_quality_score', 0),
            'rama': ep.get('mean_ramachandran_favored', 0),
            'trans': ep.get('mean_trans_peptide_fraction', 0),
            'clash': ep.get('mean_clash_rate', 0)
        })

print(f"{'Experiment':<30} {'Quality':<10} {'Rama %':<10} {'Trans %':<10} {'Clash':<10}")
print("-" * 80)
for r in sorted(results, key=lambda x: x['trans'], reverse=True):
    print(f"{r['name']:<30} {r['quality']:<10.3f} {r['rama']:<10.1%} {r['trans']:<10.1%} {r['clash']:<10.3f}")
EOF
```

---

## Time Estimates

| Configuration | Time | Use Case |
|--------------|------|----------|
| **Fast test (3 epochs, toy data)** | ~10-15 min | Quick iteration |
| **Medium test (5 epochs, full data)** | ~30-45 min | Better validation |
| **Full test (20 epochs, full data)** | ~2-4 hours | Final validation |

---

## Tips

1. **Start with fast tests** - Test many configurations quickly
2. **Focus on omega first** - This is the critical issue
3. **Test one variable at a time** - Easier to understand what works
4. **Keep notes** - Document which configurations you tested
5. **Compare systematically** - Use the comparison script above

---

## Example Workflow

```bash
# 1. Test baseline
bash test_training_fast.sh

# 2. Modify omega weight to 0.5 in enhanced_models_v2.py
# 3. Test again
bash test_training_fast.sh

# 4. Compare results
python compare_results.py

# 5. Pick best configuration
# 6. Run full training with best config
bash train_and_evaluate_advanced_flow.sh
```

---

## Troubleshooting

### Training too slow?
- Reduce `--epochs` (try 2 instead of 3)
- Reduce `--batch_size` (try 16 instead of 32)
- Disable expensive features (`--use_pairwise false`)

### Out of memory?
- Reduce `--hidden_size` (try 128 instead of 256)
- Reduce `--num_layers` (try 4 instead of 6)
- Reduce `--batch_size` (try 16)

### Results not meaningful?
- Increase `--epochs` to 5
- Use full dataset (remove `--toy`)
- Generate more samples (increase `N_SAMPLES`)

---

**Last Updated:** 2026-01-03



