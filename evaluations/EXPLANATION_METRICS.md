# Why Only Two Metrics in Summary?

## Understanding the Evaluation Summary

The `evaluation_summary.json` shows **which metric categories** were computed, not the total number of individual metrics.

### What Gets Computed

The evaluation framework computes metrics in **5 categories**:

1. ✅ **Energy/Physical Plausibility** - Always computed (no reference needed)
2. ✅ **Novelty & Diversity** - Always computed (no reference needed)
3. ⚠️ **Structural Similarity** - Requires reference structures
4. ⚠️ **Motif Recovery** - Requires reference structures + motif indices
5. ⚠️ **Sequence-Structure Compatibility** - Requires reference structures

### Why Only Two Categories in Your Summary

Looking at your `evaluation_summary.json`:
```json
{
  "n_reference_structures": 0,  // ← No reference structures provided!
  "evaluation_metrics": [
    "energy_plausibility",      // ✅ Computed
    "novelty_diversity"         // ✅ Computed
  ]
}
```

**Reason**: No reference structures were provided, so only metrics that don't require references are computed.

### What's Actually in Each Category

Even though only **2 categories** are shown, each contains **many individual metrics**:

#### 1. Energy/Physical Plausibility (15+ metrics)
- `ramachandran_favored` - Fraction in favored regions
- `ramachandran_allowed` - Fraction in allowed regions
- `ramachandran_outliers` - Fraction of outliers
- `alpha_helix_fraction` - α-helix content
- `beta_sheet_fraction` - β-sheet content
- `left_handed_alpha_fraction` - Left-handed α-helix
- `trans_peptide_fraction` - Trans peptide bonds
- `cis_peptide_fraction` - Cis peptide bonds
- `n_vdw_clashes` - Number of VDW clashes
- `clash_rate` - Clash rate per atom
- `overall_quality_score` - Composite quality score
- Plus more...

#### 2. Novelty & Diversity (10+ metrics)
- `mean_pairwise_rmsd` - Structural diversity (RMSD)
- `diversity_score` - Combined diversity score
- `mean_pairwise_tm` - Structural diversity (TM-score)
- `diversity_score_tm` - TM-based diversity
- `novel_fraction` - Fraction novel vs database
- `seq_mean_pairwise_identity` - Sequence diversity
- `seq_diversity_score` - Sequence diversity score
- Plus more...

### Enhanced Summary

The summary now includes **detailed statistics** from each category:

```json
{
  "energy_plausibility": {
    "mean_quality_score": 0.056,
    "mean_ramachandran_favored": 0.140,
    "mean_ramachandran_allowed": 0.264,
    "mean_ramachandran_outliers": 0.736,
    "mean_alpha_helix_fraction": 0.087,
    "mean_beta_sheet_fraction": 0.045,
    "mean_clash_rate": 0.XXX,
    "mean_n_vdw_clashes": XXX
  },
  "novelty_diversity": {
    "mean_pairwise_rmsd": 14.24,
    "diversity_score": 14.24,
    "seq_mean_pairwise_identity": 1.0,
    "seq_diversity_score": 0.0
  }
}
```

### To Get All 5 Categories

Provide reference structures:

```bash
python evaluations/run_evaluation.py \
    results/advanced_flow/samples_advanced_flow_260102_022234 \
    --reference_dir data/references
```

This will compute:
- ✅ Energy/Plausibility
- ✅ Novelty/Diversity
- ✅ **Structural Similarity** (RMSD, TM-score, GDT)
- ✅ **Motif Recovery** (if motifs in metadata)
- ✅ **Sequence-Structure Compatibility**

### Full Metrics Available

Check the CSV files for all computed metrics:

```bash
# See all energy/plausibility metrics
head -1 results/.../evaluation/energy_plausibility.csv

# See all diversity metrics
head -1 results/.../evaluation/novelty_diversity.csv
```

Each CSV file contains **all individual metrics** for each structure, while the summary provides **aggregated statistics** across all structures.

## Summary

- **2 categories** shown = Only metrics that don't need references
- **Many metrics per category** = Each CSV has 10-15+ individual metrics
- **Enhanced summary** = Now shows detailed statistics from each category
- **To get all 5 categories** = Provide `--reference_dir`

The evaluation is working correctly - it computes all available metrics based on what's provided!

