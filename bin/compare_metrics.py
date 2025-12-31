#!/usr/bin/env python3
"""
Comprehensive metric comparison across all generated samples.

Compares:
1. Angle distributions (Ramachandran-like analysis)
2. Diversity metrics
3. Structural validity
4. Motif preservation
5. Generation quality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_scenario_samples(scenario_dir: Path):
    """Load all samples from a scenario"""
    angles_dir = scenario_dir / "angles"
    if not angles_dir.exists():
        return None
    
    samples = []
    for f in sorted(angles_dir.glob("*.csv")):
        angles = np.loadtxt(f, delimiter=",", skiprows=1)
        samples.append(angles)
    
    return np.array(samples)


def compute_angle_statistics(samples: np.ndarray):
    """Compute comprehensive angle statistics"""
    n_samples, seq_len, n_angles = samples.shape
    
    stats_dict = {
        'n_samples': n_samples,
        'seq_len': seq_len,
        'mean': samples.mean(axis=(0, 1)),
        'std': samples.std(axis=(0, 1)),
        'median': np.median(samples, axis=(0, 1)),
        'q25': np.percentile(samples, 25, axis=(0, 1)),
        'q75': np.percentile(samples, 75, axis=(0, 1)),
        'min': samples.min(axis=(0, 1)),
        'max': samples.max(axis=(0, 1)),
    }
    
    return stats_dict


def compute_ramachandran_stats(samples: np.ndarray):
    """Analyze phi-psi distribution (Ramachandran-like)"""
    phi = samples[:, :, 0].flatten()
    psi = samples[:, :, 1].flatten()
    
    # Remove zeros (padding/motif regions)
    mask = (phi != 0) | (psi != 0)
    phi = phi[mask]
    psi = psi[mask]
    
    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(phi, psi, bins=50, 
                                        range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    
    # Identify favored regions (simplified Ramachandran)
    # Alpha helix: phi ~ -60°, psi ~ -45°
    # Beta sheet: phi ~ -120°, psi ~ 120°
    alpha_mask = (phi > -2.0) & (phi < -0.5) & (psi > -1.5) & (psi < 0)
    beta_mask = (phi > -2.5) & (phi < -1.5) & (psi > 1.5) & (psi < 2.5)
    
    stats_dict = {
        'n_residues': len(phi),
        'phi_mean': phi.mean(),
        'phi_std': phi.std(),
        'psi_mean': psi.mean(),
        'psi_std': psi.std(),
        'alpha_fraction': alpha_mask.sum() / len(phi),
        'beta_fraction': beta_mask.sum() / len(phi),
        'histogram': H,
        'phi_edges': xedges,
        'psi_edges': yedges,
    }
    
    return stats_dict


def compute_diversity_metrics(samples: np.ndarray):
    """Compute diversity across samples"""
    n_samples, seq_len, n_angles = samples.shape
    
    # Flatten samples for pairwise comparison
    samples_flat = samples.reshape(n_samples, -1)
    
    # Pairwise distances
    distances = pdist(samples_flat, metric='euclidean')
    
    # Self-similarity (lower is more diverse)
    self_similarity = 1.0 / (1.0 + distances.mean())
    
    # Coefficient of variation (higher is more diverse)
    cv = samples.std(axis=0).mean() / (np.abs(samples.mean(axis=0)).mean() + 1e-8)
    
    # Entropy of angle distributions
    entropies = []
    for angle_idx in range(n_angles):
        angle_data = samples[:, :, angle_idx].flatten()
        angle_data = angle_data[angle_data != 0]  # Remove padding
        if len(angle_data) > 0:
            hist, _ = np.histogram(angle_data, bins=50, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            entropy = -np.sum(hist * np.log(hist))
            entropies.append(entropy)
    
    diversity_dict = {
        'mean_pairwise_distance': distances.mean(),
        'std_pairwise_distance': distances.std(),
        'min_pairwise_distance': distances.min(),
        'max_pairwise_distance': distances.max(),
        'self_similarity': self_similarity,
        'coefficient_of_variation': cv,
        'mean_entropy': np.mean(entropies),
        'distance_matrix': squareform(distances),
    }
    
    return diversity_dict


def compute_structural_validity(samples: np.ndarray):
    """Check structural validity metrics"""
    n_samples, seq_len, n_angles = samples.shape
    
    # Check for NaN or Inf
    has_nan = np.isnan(samples).any()
    has_inf = np.isinf(samples).any()
    
    # Check angle ranges (should be roughly [-pi, pi] for dihedrals)
    phi_psi_omega = samples[:, :, :3]  # First 3 are dihedrals
    out_of_range = (np.abs(phi_psi_omega) > 2 * np.pi).sum()
    
    # Check for constant regions (might indicate issues)
    constant_residues = 0
    for i in range(n_samples):
        for j in range(seq_len):
            if np.allclose(samples[i, j, :], 0):
                constant_residues += 1
    
    # Bond angle validity (tau should be around 110° = 1.92 rad)
    tau = samples[:, :, 3]
    tau_valid = tau[tau != 0]
    tau_mean = tau_valid.mean() if len(tau_valid) > 0 else 0
    tau_std = tau_valid.std() if len(tau_valid) > 0 else 0
    
    validity_dict = {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'out_of_range_count': int(out_of_range),
        'out_of_range_fraction': out_of_range / (n_samples * seq_len * 3),
        'constant_residues': constant_residues,
        'constant_fraction': constant_residues / (n_samples * seq_len),
        'tau_mean': float(tau_mean),
        'tau_std': float(tau_std),
        'tau_expected': 1.92,  # ~110 degrees
        'tau_deviation': abs(float(tau_mean) - 1.92) if tau_mean != 0 else 0,
    }
    
    return validity_dict


def compute_motif_preservation(samples: np.ndarray, motif_regions: list):
    """Check if motif regions are preserved (should be zeros in our case)"""
    if not motif_regions:
        return {'has_motifs': False}
    
    n_samples, seq_len, n_angles = samples.shape
    
    # Check if motif regions are zeros
    motif_preserved = []
    for start, end in motif_regions:
        if end <= seq_len:
            motif_angles = samples[:, start:end, :]
            is_zero = np.allclose(motif_angles, 0)
            motif_preserved.append(is_zero)
    
    preservation_dict = {
        'has_motifs': True,
        'n_motifs': len(motif_regions),
        'motif_regions': motif_regions,
        'all_preserved': all(motif_preserved),
        'preservation_rate': sum(motif_preserved) / len(motif_preserved) if motif_preserved else 0,
    }
    
    return preservation_dict


def plot_ramachandran(rama_stats: dict, scenario_name: str, output_dir: Path):
    """Plot Ramachandran-like plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    H = rama_stats['histogram']
    extent = [rama_stats['phi_edges'][0], rama_stats['phi_edges'][-1],
              rama_stats['psi_edges'][0], rama_stats['psi_edges'][-1]]
    
    im = ax.imshow(H.T, origin='lower', extent=extent, aspect='auto', 
                   cmap='viridis', interpolation='bilinear')
    
    ax.set_xlabel('Phi (radians)', fontsize=12)
    ax.set_ylabel('Psi (radians)', fontsize=12)
    ax.set_title(f'Ramachandran Plot - {scenario_name}', fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.colorbar(im, ax=ax, label='Density')
    plt.tight_layout()
    plt.savefig(output_dir / f'ramachandran_{scenario_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_diversity_heatmap(diversity_stats: dict, scenario_name: str, output_dir: Path):
    """Plot pairwise distance heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    dist_matrix = diversity_stats['distance_matrix']
    
    im = ax.imshow(dist_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'Sample Diversity - {scenario_name}', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Pairwise Distance')
    plt.tight_layout()
    plt.savefig(output_dir / f'diversity_{scenario_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_angle_distributions(samples: np.ndarray, scenario_name: str, output_dir: Path):
    """Plot angle distributions"""
    angle_names = ['Phi', 'Psi', 'Omega', 'Tau', 'CA:C:1N', 'C:1N:1CA']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        angle_data = samples[:, :, i].flatten()
        angle_data = angle_data[angle_data != 0]  # Remove padding
        
        if len(angle_data) > 0:
            ax.hist(angle_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(angle_data.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {angle_data.mean():.2f}')
            ax.set_xlabel(f'{name} (radians)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{name} Distribution', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.suptitle(f'Angle Distributions - {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'angle_distributions_{scenario_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_table(all_results: dict, output_dir: Path):
    """Create comparison table across scenarios"""
    
    # Prepare data for table
    rows = []
    for scenario, results in all_results.items():
        row = {
            'Scenario': scenario,
            'N Samples': results['angle_stats']['n_samples'],
            'Seq Length': results['angle_stats']['seq_len'],
            'Phi Mean': f"{results['angle_stats']['mean'][0]:.3f}",
            'Psi Mean': f"{results['angle_stats']['mean'][1]:.3f}",
            'Phi Std': f"{results['angle_stats']['std'][0]:.3f}",
            'Psi Std': f"{results['angle_stats']['std'][1]:.3f}",
            'Diversity': f"{results['diversity']['mean_pairwise_distance']:.2f}",
            'Alpha %': f"{results['ramachandran']['alpha_fraction']*100:.1f}",
            'Beta %': f"{results['ramachandran']['beta_fraction']*100:.1f}",
            'Valid': '✓' if not results['validity']['has_nan'] else '✗',
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(output_dir / 'comparison_table.csv', index=False)
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Comparison Across All Scenarios', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return df


def create_summary_plots(all_results: dict, output_dir: Path):
    """Create summary comparison plots"""
    
    scenarios = list(all_results.keys())
    
    # 1. Diversity comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    diversities = [all_results[s]['diversity']['mean_pairwise_distance'] for s in scenarios]
    cvs = [all_results[s]['diversity']['coefficient_of_variation'] for s in scenarios]
    
    axes[0].bar(range(len(scenarios)), diversities, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(scenarios)))
    axes[0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Pairwise Distance', fontsize=11)
    axes[0].set_title('Sample Diversity', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(range(len(scenarios)), cvs, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(scenarios)))
    axes[1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[1].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[1].set_title('Angle Variability', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Secondary structure comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alpha_fracs = [all_results[s]['ramachandran']['alpha_fraction'] * 100 for s in scenarios]
    beta_fracs = [all_results[s]['ramachandran']['beta_fraction'] * 100 for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax.bar(x - width/2, alpha_fracs, width, label='Alpha-like', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, beta_fracs, width, label='Beta-like', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Secondary Structure Content', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'secondary_structure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Angle statistics comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    angle_names = ['Phi', 'Psi', 'Omega', 'Tau', 'CA:C:1N', 'C:1N:1CA']
    
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        means = [all_results[s]['angle_stats']['mean'][i] for s in scenarios]
        stds = [all_results[s]['angle_stats']['std'][i] for s in scenarios]
        
        x = np.arange(len(scenarios))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'{name} (rad)', fontsize=10)
        ax.set_title(f'{name} Mean ± Std', fontsize=11, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Angle Statistics Across Scenarios', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'angle_stats_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path("results/samples/production_v1")
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = {
        "short_single_motif": [(10, 20)],
        "medium_two_motifs": [(10, 20), (50, 60)],
        "long_single_motif": [(60, 80)],
        "unconditional": [],
    }
    
    print("="*80)
    print("COMPREHENSIVE METRIC COMPARISON")
    print("="*80)
    
    all_results = {}
    
    for scenario_name, motif_regions in scenarios.items():
        print(f"\nAnalyzing: {scenario_name}")
        print("-" * 80)
        
        scenario_dir = base_dir / scenario_name
        samples = load_scenario_samples(scenario_dir)
        
        if samples is None:
            print(f"  ⚠️  No samples found")
            continue
        
        print(f"  ✓ Loaded {len(samples)} samples")
        
        # Compute all metrics
        angle_stats = compute_angle_statistics(samples)
        rama_stats = compute_ramachandran_stats(samples)
        diversity_stats = compute_diversity_metrics(samples)
        validity_stats = compute_structural_validity(samples)
        motif_stats = compute_motif_preservation(samples, motif_regions)
        
        # Store results
        all_results[scenario_name] = {
            'angle_stats': angle_stats,
            'ramachandran': rama_stats,
            'diversity': diversity_stats,
            'validity': validity_stats,
            'motif': motif_stats,
        }
        
        # Print summary
        print(f"  Angle Stats: mean phi={angle_stats['mean'][0]:.3f}, psi={angle_stats['mean'][1]:.3f}")
        print(f"  Diversity: {diversity_stats['mean_pairwise_distance']:.2f} ± {diversity_stats['std_pairwise_distance']:.2f}")
        print(f"  Secondary Structure: α={rama_stats['alpha_fraction']*100:.1f}%, β={rama_stats['beta_fraction']*100:.1f}%")
        print(f"  Validity: {'✓ Valid' if not validity_stats['has_nan'] else '✗ Invalid'}")
        
        # Generate plots
        plot_ramachandran(rama_stats, scenario_name, output_dir)
        plot_diversity_heatmap(diversity_stats, scenario_name, output_dir)
        plot_angle_distributions(samples, scenario_name, output_dir)
    
    # Create comparison visualizations
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    comparison_df = create_comparison_table(all_results, output_dir)
    create_summary_plots(all_results, output_dir)
    
    # Save detailed results as JSON
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    results_json = {}
    for scenario, results in all_results.items():
        results_json[scenario] = {
            'angle_stats': {k: convert_to_json_serializable(v) 
                           for k, v in results['angle_stats'].items()},
            'ramachandran': {k: convert_to_json_serializable(v) 
                            for k, v in results['ramachandran'].items() 
                            if k not in ['histogram', 'phi_edges', 'psi_edges']},
            'diversity': {k: convert_to_json_serializable(v) 
                         for k, v in results['diversity'].items() 
                         if k != 'distance_matrix'},
            'validity': convert_to_json_serializable(results['validity']),
            'motif': convert_to_json_serializable(results['motif']),
        }
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - comparison_table.csv/png")
    print("  - diversity_comparison.png")
    print("  - secondary_structure_comparison.png")
    print("  - angle_stats_comparison.png")
    print("  - ramachandran_*.png (per scenario)")
    print("  - diversity_*.png (per scenario)")
    print("  - angle_distributions_*.png (per scenario)")
    print("  - detailed_results.json")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Print comparison table
    print("\n" + comparison_df.to_string(index=False))
    
    # Highlight best/worst
    print("\n" + "="*80)
    print("HIGHLIGHTS")
    print("="*80)
    
    diversities = {s: all_results[s]['diversity']['mean_pairwise_distance'] 
                   for s in all_results.keys()}
    most_diverse = max(diversities, key=diversities.get)
    least_diverse = min(diversities, key=diversities.get)
    
    print(f"\n✓ Most Diverse: {most_diverse} (distance: {diversities[most_diverse]:.2f})")
    print(f"✓ Least Diverse: {least_diverse} (distance: {diversities[least_diverse]:.2f})")
    
    alpha_fracs = {s: all_results[s]['ramachandran']['alpha_fraction'] * 100 
                   for s in all_results.keys()}
    most_alpha = max(alpha_fracs, key=alpha_fracs.get)
    
    print(f"\n✓ Most Alpha-like: {most_alpha} ({alpha_fracs[most_alpha]:.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
