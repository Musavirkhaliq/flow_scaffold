# Protein Generative Model Evaluation Framework

Comprehensive evaluation framework for protein generative models, specifically designed for **motif scaffolding** and **flow matching** approaches using CATH data.

## Overview

This evaluation framework provides a comprehensive suite of metrics to assess protein generative models across multiple dimensions:

1. **Structural Similarity** - RMSD, TM-score, GDT
2. **Motif Recovery** - Local motif preservation accuracy
3. **Sequence-Structure Compatibility** - Sequence recovery and similarity
4. **Energy/Physical Plausibility** - Rosetta energy, AlphaFold2 confidence, VDW clashes, Ramachandran quality
5. **Novelty & Diversity** - Structural and sequence diversity, novelty vs. databases

## Installation

The evaluation framework uses existing project dependencies. Ensure you have:

- `biotite` - For PDB structure handling
- `scipy` - For structural alignment
- `pandas` - For data handling
- `numpy` - For numerical operations

Optional dependencies:
- **TMalign** - For TM-score computation (must be in PATH)
- **Rosetta** - For energy calculations (optional)
- **AlphaFold2** - For confidence scores (optional)

## Quick Start

### Basic Evaluation

```bash
python evaluations/evaluate_pipeline.py \
    --generated_dir results/samples \
    --reference_dir data/references \
    --output_dir evaluation_results
```

### With Motif Indices

```bash
python evaluations/evaluate_pipeline.py \
    --generated_dir results/samples \
    --reference_dir data/references \
    --motif_indices "10-20,30-40" \
    --output_dir evaluation_results
```

### Novelty Evaluation with Database

```bash
python evaluations/evaluate_pipeline.py \
    --generated_dir results/samples \
    --database_dir data/cath \
    --output_dir evaluation_results
```

## Evaluation Metrics

### 1. Structural Similarity

**Metrics:**
- **RMSD** (Root Mean Square Deviation) - Overall structural similarity
- **TM-score** (Template Modeling Score) - Topological similarity (0-1, higher is better)
- **GDT** (Global Distance Test) - Fraction of residues within distance thresholds
- **Motif RMSD** - RMSD specifically for motif regions
- **Motif TM-score** - TM-score for motif regions

**Usage:**
```python
from evaluations.structural_similarity import evaluate_structural_similarity

results = evaluate_structural_similarity(
    query_pdb="generated.pdb",
    reference_pdb="reference.pdb",
    motif_indices=[10, 11, 12, 13, 14]
)
```

### 2. Motif Recovery

**Metrics:**
- **Motif RMSD** - RMSD of motif region after optimal superposition
- **Motif TM-score** - TM-score of motif region
- **Motif Preservation** - Binary indicator (preserved if RMSD < threshold and TM > 0.5)
- **Interface Quality** - Quality of scaffold-motif interface region
- **Per-residue distances** - Distance statistics for motif residues

**Usage:**
```python
from evaluations.motif_recovery import evaluate_motif_recovery

results = evaluate_motif_recovery(
    query_pdb="generated.pdb",
    reference_pdb="reference.pdb",
    motif_indices=[10, 11, 12, 13, 14],
    threshold_rmsd=2.0
)
```

### 3. Sequence-Structure Compatibility

**Metrics:**
- **Exact Recovery Rate** - Percentage of residues matching exactly
- **Similar Recovery Rate** - Percentage matching with BLOSUM62-like similarity
- **Property Recovery Rate** - Percentage matching in amino acid properties
- **Sequence Identity** - Overall sequence identity
- **Sequence Similarity** - BLOSUM62-based similarity
- **Hamming Distance** - Number of mismatches

**Usage:**
```python
from evaluations.sequence_structure_compatibility import evaluate_sequence_structure_compatibility

results = evaluate_sequence_structure_compatibility(
    generated_pdb="generated.pdb",
    reference_pdb="reference.pdb"
)
```

### 4. Energy/Physical Plausibility

**Metrics:**
- **Ramachandran Quality** - Fraction of residues in favored/allowed regions
- **VDW Clash Score** - Number and rate of van der Waals clashes
- **Rosetta Energy** - Total energy, attractive, repulsive, solvation, electrostatic (if available)
- **AlphaFold2 Confidence** - pLDDT scores (if available)
- **Overall Quality Score** - Composite quality metric

**Usage:**
```python
from evaluations.energy_plausibility import evaluate_energy_plausibility

results = evaluate_energy_plausibility(
    pdb_path="structure.pdb",
    compute_rosetta=True,
    compute_alphafold=False
)
```

### 5. Novelty & Diversity

**Metrics:**
- **Structural Diversity** - Pairwise RMSD/TM-score statistics
- **Sequence Diversity** - Position-wise entropy, pairwise identity
- **Novelty vs Database** - Maximum TM-score against reference database
- **Novel Fraction** - Fraction of structures with TM-score < 0.5 vs database

**Usage:**
```python
from evaluations.novelty_diversity import evaluate_novelty_diversity

results = evaluate_novelty_diversity(
    generated_pdbs=["gen1.pdb", "gen2.pdb", ...],
    database_pdbs=["ref1.pdb", "ref2.pdb", ...]  # Optional
)
```

## Module Structure

```
evaluations/
├── __init__.py                          # Package initialization
├── README.md                            # This file
├── evaluate_pipeline.py                # Main evaluation pipeline
├── structural_similarity.py            # RMSD, TM-score, GDT
├── motif_recovery.py                    # Motif-specific metrics
├── sequence_structure_compatibility.py  # Sequence metrics
├── energy_plausibility.py               # Energy/quality metrics
└── novelty_diversity.py                 # Diversity/novelty metrics
```

## Academic Benchmarks

This framework is designed to be compatible with standard benchmarks:

- **MotifBench** - 30 motif-scaffolding problems with standardized metrics
- **Protein-SE(3)** - Unified framework for SE(3)-based generative models
- **CATH** - Database for novelty evaluation

## Output Format

The evaluation pipeline generates:

1. **CSV files** - One per metric category:
   - `structural_similarity.csv`
   - `motif_recovery.csv`
   - `sequence_structure_compatibility.csv`
   - `energy_plausibility.csv`
   - `novelty_diversity.csv`

2. **JSON summary** - `evaluation_summary.json` with overall statistics

## Advanced Usage

### Batch Evaluation

```python
from evaluations.structural_similarity import batch_evaluate_structural_similarity

results_df = batch_evaluate_structural_similarity(
    query_pdbs=["gen1.pdb", "gen2.pdb"],
    reference_pdbs=["ref1.pdb", "ref2.pdb"],
    motif_indices_list=[[10, 11, 12], [20, 21, 22]],
    n_threads=8
)
```

### Custom Evaluation

```python
from evaluations import (
    compute_rmsd_from_pdb,
    compute_tm_score,
    compute_ramachandran_quality,
    compute_vdw_clash_score
)

# Individual metrics
rmsd = compute_rmsd_from_pdb("gen.pdb", "ref.pdb")
tm = compute_tm_score("gen.pdb", "ref.pdb")
ramachandran = compute_ramachandran_quality("gen.pdb")
clashes = compute_vdw_clash_score("gen.pdb")
```

## Performance Notes

- **Parallel Processing**: Most batch functions support parallel processing via `n_threads`
- **TMalign**: TM-score computation can be slow; use `fast=True` for faster (less accurate) results
- **Rosetta**: Energy calculation is computationally expensive
- **Database Comparison**: Novelty evaluation scales with database size; use `max_comparisons` to limit

## Troubleshooting

### TMalign Not Found
```bash
# Install TMalign and add to PATH
# Download from: https://zhanggroup.org/TM-align/
export PATH=$PATH:/path/to/TMalign
```

### Rosetta Not Available
Rosetta energy calculation is optional. The evaluation will skip it if not found.

### Memory Issues
For large datasets, process in batches or reduce `n_threads`.

## Citation

If you use this evaluation framework, please cite:

- **TM-score**: Zhang & Skolnick (2004) "Scoring function for automated assessment of protein structure template quality"
- **GDT**: Zemla (2003) "LGA: A method for finding 3D similarities in protein structures"
- **MotifBench**: [Reference when published]
- **Protein-SE(3)**: [Reference when published]

## License

This evaluation framework is part of the flow_scaffold project.

