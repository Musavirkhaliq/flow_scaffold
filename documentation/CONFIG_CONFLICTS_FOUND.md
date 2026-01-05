# Config Conflicts Found - CRITICAL ISSUES

**Date:** 2026-01-05  
**Status:** 🔴 **CRITICAL CONFLICTS IDENTIFIED**

---

## Critical Conflicts Between Config and Code Defaults

### 1. ❌ **BATCH_SIZE Conflict** (CRITICAL)

**Config (`config_advanced_flow.sh`):**
```bash
BATCH_SIZE=16
```

**Code Default (`bin/train_advanced_flow.py` line 102):**
```python
parser.add_argument("--batch_size", type=int, default=32)
```

**Problem:** 
- If config value isn't passed correctly, code will use 32 instead of 16
- Training script passes `${BATCH_SIZE}` so this should be OK, but default is wrong

**Impact:** Wrong batch size if config isn't sourced properly

---

### 2. ❌ **EPOCHS Conflict** (CRITICAL)

**Config (`config_advanced_flow.sh`):**
```bash
EPOCHS=150
```

**Code Default (`bin/train_advanced_flow.py` line 106):**
```python
parser.add_argument("--epochs", type=int, default=50)
```

**Problem:**
- Config says 150 epochs, but code default is 50
- If config isn't passed, training will stop at 50 epochs instead of 150

**Impact:** Model will be undertrained (50 epochs instead of 150)

---

### 3. ❌ **LR_SCHEDULER Conflict** (CRITICAL)

**Config (`config_advanced_flow.sh`):**
```bash
LR_SCHEDULER="ReduceLROnPlateau"
```

**Code Default (`bin/train_advanced_flow.py` line 107):**
```python
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealing", ...)
```

**Problem:**
- Config says ReduceLROnPlateau, but code default is CosineAnnealing
- If config isn't passed, wrong scheduler will be used

**Impact:** Validation loss plateau won't be handled correctly

---

### 4. ❌ **GEOMETRIC_WEIGHT Conflict** (CRITICAL)

**Config (`config_advanced_flow.sh`):**
```bash
GEOMETRIC_WEIGHT=0.20
```

**Code Default (`bin/train_advanced_flow.py` line 82):**
```python
parser.add_argument("--geometric_weight", type=float, default=0.15)
```

**Problem:**
- Config says 0.20, but code default is 0.15
- If config isn't passed, weaker geometric constraints will be used

**Impact:** Ramachandran quality will be worse (weaker constraints)

---

### 5. ❌ **ACCUMULATE_GRAD_BATCHES Conflict** (MODERATE)

**Config (`config_advanced_flow.sh`):**
```bash
ACCUMULATE_GRAD_BATCHES=2
```

**Code Default (`bin/train_advanced_flow.py` line 103):**
```python
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
```

**Problem:**
- Config says 2, but code default is 1
- If config isn't passed, effective batch size will be wrong

**Impact:** Different effective batch size than intended

---

## Verification: Are Config Values Actually Passed?

### ✅ **Training Script (`phase1_train_advanced_flow.sh`)**

The training script DOES pass config values:
- Line 139: `--batch_size ${BATCH_SIZE}` ✅
- Line 142: `--epochs ${EPOCHS}` ✅
- Line 143: `--lr_scheduler ${LR_SCHEDULER}` ✅
- Line 199: `--geometric_weight ${GEOMETRIC_WEIGHT}` ✅
- Line 140: `--accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES}` ✅

**Status:** Config values ARE passed, so conflicts only matter if:
1. Script is run directly without sourcing config
2. Config file isn't sourced properly
3. Environment variables are overridden

---

## Recommended Fixes

### Fix 1: Update Code Defaults to Match Config (SAFEST)

Update argument parser defaults to match config values:

```python
# In bin/train_advanced_flow.py
parser.add_argument("--batch_size", type=int, default=16)  # Match config
parser.add_argument("--epochs", type=int, default=150)  # Match config
parser.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau", ...)  # Match config
parser.add_argument("--geometric_weight", type=float, default=0.20)  # Match config
parser.add_argument("--accumulate_grad_batches", type=int, default=2)  # Match config
```

**Why:** If config isn't sourced, code will still use correct defaults

---

### Fix 2: Add Validation in Training Script (SAFER)

Add checks to ensure config is sourced:

```bash
# In phase1_train_advanced_flow.sh, after sourcing config
if [ -z "$BATCH_SIZE" ] || [ -z "$EPOCHS" ]; then
    echo "ERROR: Config not sourced properly!"
    echo "Please run: source config_advanced_flow.sh"
    exit 1
fi
```

**Why:** Catches errors early if config isn't loaded

---

### Fix 3: Use Config File Directly (BEST)

Instead of relying on environment variables, read config file directly in Python:

```python
# In bin/train_advanced_flow.py
import configparser
import os

# Read config file
config_file = os.path.join(os.path.dirname(__file__), '..', 'config_advanced_flow.sh')
# Parse and use values as defaults
```

**Why:** Most robust - doesn't depend on environment

---

## Priority

### Critical (Fix Immediately):
1. ✅ Update `--epochs` default: 50 → 150
2. ✅ Update `--lr_scheduler` default: CosineAnnealing → ReduceLROnPlateau
3. ✅ Update `--geometric_weight` default: 0.15 → 0.20

### Important (Fix Soon):
4. ✅ Update `--batch_size` default: 32 → 16
5. ✅ Update `--accumulate_grad_batches` default: 1 → 2

---

## Impact if Not Fixed

If config isn't sourced properly:
- **Epochs:** 150 → 50 (model undertrained)
- **LR Scheduler:** ReduceLROnPlateau → CosineAnnealing (validation loss plateau won't be handled)
- **Geometric Weight:** 0.20 → 0.15 (weaker Ramachandran constraints)
- **Batch Size:** 16 → 32 (different training dynamics)
- **Gradient Accumulation:** 2 → 1 (different effective batch size)

---

**Status:** 🔴 **CRITICAL - FIX DEFAULT VALUES TO MATCH CONFIG**

