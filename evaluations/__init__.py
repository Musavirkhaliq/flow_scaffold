"""
Comprehensive evaluation framework for protein generative models.

This package provides evaluation metrics for:
- Structural similarity (RMSD, TM-score, GDT)
- Motif recovery accuracy
- Sequence-structure compatibility
- Energy/physical plausibility (Rosetta, AlphaFold2)
- Novelty and diversity metrics
"""

from evaluations.structural_similarity import (
    compute_rmsd,
    compute_rmsd_from_pdb,
    compute_tm_score,
    compute_gdt,
    compute_motif_rmsd,
    compute_motif_tm_score,
    evaluate_structural_similarity,
    batch_evaluate_structural_similarity,
)

from evaluations.motif_recovery import (
    compute_motif_superposition_accuracy,
    compute_motif_preservation_score,
    compute_scaffold_motif_interface_quality,
    evaluate_motif_recovery,
    batch_evaluate_motif_recovery,
)

from evaluations.sequence_structure_compatibility import (
    extract_sequence_from_pdb,
    compute_sequence_recovery_rate,
    compute_sequence_similarity,
    compute_sequence_diversity,
    evaluate_sequence_structure_compatibility,
    batch_evaluate_sequence_structure_compatibility,
)

from evaluations.energy_plausibility import (
    compute_ramachandran_quality,
    compute_vdw_clash_score,
    compute_rosetta_energy,
    compute_alphafold2_confidence,
    evaluate_energy_plausibility,
    batch_evaluate_energy_plausibility,
)

from evaluations.novelty_diversity import (
    compute_structural_diversity,
    compute_structural_diversity_tm_score,
    compute_novelty_vs_database,
    compute_sequence_diversity_metrics,
    evaluate_novelty_diversity,
    batch_evaluate_novelty_diversity,
)

__all__ = [
    # Structural similarity
    "compute_rmsd",
    "compute_rmsd_from_pdb",
    "compute_tm_score",
    "compute_gdt",
    "compute_motif_rmsd",
    "compute_motif_tm_score",
    "evaluate_structural_similarity",
    "batch_evaluate_structural_similarity",
    # Motif recovery
    "compute_motif_superposition_accuracy",
    "compute_motif_preservation_score",
    "compute_scaffold_motif_interface_quality",
    "evaluate_motif_recovery",
    "batch_evaluate_motif_recovery",
    # Sequence-structure compatibility
    "extract_sequence_from_pdb",
    "compute_sequence_recovery_rate",
    "compute_sequence_similarity",
    "compute_sequence_diversity",
    "evaluate_sequence_structure_compatibility",
    "batch_evaluate_sequence_structure_compatibility",
    # Energy/plausibility
    "compute_ramachandran_quality",
    "compute_vdw_clash_score",
    "compute_rosetta_energy",
    "compute_alphafold2_confidence",
    "evaluate_energy_plausibility",
    "batch_evaluate_energy_plausibility",
    # Novelty/diversity
    "compute_structural_diversity",
    "compute_structural_diversity_tm_score",
    "compute_novelty_vs_database",
    "compute_sequence_diversity_metrics",
    "evaluate_novelty_diversity",
    "batch_evaluate_novelty_diversity",
]

