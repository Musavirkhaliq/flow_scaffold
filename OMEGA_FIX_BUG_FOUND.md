# Omega Fix Bug Found and Fixed

**Date:** 2026-01-02  
**Issue:** Omega fix was not working - 99.9% cis instead of trans

---

## The Bug

### Problem:

The omega fix was applied **before** mean correction, causing omega to wrap incorrectly:

1. **Omega fix sets omega = π** (line 647) ✓
2. **Mean correction adds omega mean (~π)** (line 730) → omega = π + π = 2π
3. **Wrapping converts 2π → 0** (cis) ✗

### Root Cause:

**Order of operations was wrong:**
- Fix omega to π → Add mean (~π) → Wrap → omega = 0 (cis) ✗

**Should be:**
- Add mean (~π) → omega = 0 + π = π → Fix to π → omega = π (trans) ✓

---

## The Fix

### Solution:

Move omega fix to **AFTER** mean correction:

```python
# Before (WRONG):
sample[:, 2] = np.pi  # Fix omega
# ... later ...
sample_corrected = apply_means_with_wrapping(sample, means)  # Adds mean → wraps to 0

# After (CORRECT):
sample_corrected = apply_means_with_wrapping(sample, means)  # Adds mean
sample_corrected[:, 2] = np.pi  # Fix omega AFTER mean correction
```

### Why This Works:

1. **Model learned mean-centered angles:** omega ~0 (after mean centering)
2. **Omega mean is ~π:** So after adding mean, omega = 0 + π = π (trans) ✓
3. **Explicitly set to π:** Ensures omega is exactly π, avoiding any wrapping issues

---

## Expected Impact

### Before Fix:
- Trans Peptide Fraction: **0.1%** (should be >95%)
- Cis Peptide Fraction: **99.9%** (should be <5%)
- Clash Rate: **0.616** (should be <0.1)
- Quality Score: **0.100** (should be >0.7)

### After Fix:
- Trans Peptide Fraction: **>95%** (950x improvement expected)
- Cis Peptide Fraction: **<5%** (20x improvement expected)
- Clash Rate: **0.2-0.4** (2-3x improvement expected)
- Quality Score: **0.3-0.5** (3-5x improvement expected)

---

## Files Modified

1. **`bin/sample_advanced_flow.py`**
   - Moved omega fix from line 647 (before mean correction) to line 735 (after mean correction)
   - Added explanation comments

---

## Next Steps

1. **Re-sample structures** with the fix applied
2. **Re-evaluate** to verify omega is now >95% trans
3. **Verify improvements** in clash rate and quality score

---

## Conclusion

**The omega fix bug has been identified and fixed.**

The issue was the **order of operations** - omega was fixed before mean correction, causing it to wrap incorrectly. Moving the fix to after mean correction should resolve the 99.9% cis peptide bond issue.

**Expected result:** >95% trans peptide bonds, leading to significant improvement in clash rate and overall quality.

