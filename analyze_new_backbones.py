#!/usr/bin/env python3
"""
Analyze backbone diversity in newly generated structures
"""

import numpy as np
from pathlib import Path
import json

def parse_pdb(pdb_file):
    """Extract CA coordinates from PDB file"""
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def compute_ca_distances(coords):
    """Compute consecutive CA-CA distances"""
    if len(coords) < 2:
        return np.array([])
    dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    return dists

def compute_angles(coords):
    """Compute CA-CA-CA angles"""
    if len(coords) < 3:
        return np.array([])
    
    angles = []
    for i in range(len(coords) - 2):
        v1 = coords[i+1] - coords[i]
        v2 = coords[i+2] - coords[i+1]
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    
    return np.array(angles)

def compute_dihedrals(coords):
    """Compute CA-CA-CA-CA dihedral angles"""
    if len(coords) < 4:
        return np.array([])
    
    dihedrals = []
    for i in range(len(coords) - 3):
        b1 = coords[i+1] - coords[i]
        b2 = coords[i+2] - coords[i+1]
        b3 = coords[i+3] - coords[i+2]
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1_norm = n1 / (np.linalg.norm(n1) + 1e-8)
        n2_norm = n2 / (np.linalg.norm(n2) + 1e-8)
        
        cos_angle = np.clip(np.dot(n1_norm, n2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Determine sign
        if np.dot(np.cross(n1, n2), b2) < 0:
            angle = -angle
        
        dihedrals.append(np.degrees(angle))
    
    return np.array(dihedrals)

def analyze_structure(pdb_file):
    """Analyze a single structure"""
    coords = parse_pdb(pdb_file)
    
    if len(coords) < 4:
        return None
    
    distances = compute_ca_distances(coords)
    angles = compute_angles(coords)
    dihedrals = compute_dihedrals(coords)
    
    return {
        'length': len(coords),
        'distances': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances))
        },
        'angles': {
            'mean': float(np.mean(angles)),
            'std': float(np.std(angles)),
            'min': float(np.min(angles)),
            'max': float(np.max(angles))
        },
        'dihedrals': {
            'mean': float(np.mean(dihedrals)),
            'std': float(np.std(dihedrals)),
            'min': float(np.min(dihedrals)),
            'max': float(np.max(dihedrals))
        }
    }

def main():
    pdb_dir = Path("flow_scaf/results/enhanced_flow/samples_enhanced_flow_251218_222851/unconditional/pdb")
    
    if not pdb_dir.exists():
        print(f"Error: Directory not found: {pdb_dir}")
        return
    
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    
    if not pdb_files:
        print(f"Error: No PDB files found in {pdb_dir}")
        return
    
    print("=" * 80)
    print("BACKBONE DIVERSITY ANALYSIS - NEW SAMPLES")
    print("=" * 80)
    print(f"\nAnalyzing {len(pdb_files)} structures from:")
    print(f"  {pdb_dir}")
    print()
    
    all_stats = []
    
    for pdb_file in pdb_files[:10]:  # Analyze first 10
        stats = analyze_structure(pdb_file)
        if stats:
            all_stats.append(stats)
            print(f"\n{pdb_file.name}:")
            print(f"  Length: {stats['length']} residues")
            print(f"  CA-CA distances: {stats['distances']['mean']:.3f} ± {stats['distances']['std']:.3f} Å")
            print(f"    Range: [{stats['distances']['min']:.3f}, {stats['distances']['max']:.3f}]")
            print(f"  CA-CA-CA angles: {stats['angles']['mean']:.1f} ± {stats['angles']['std']:.1f}°")
            print(f"    Range: [{stats['angles']['min']:.1f}, {stats['angles']['max']:.1f}]")
            print(f"  Dihedrals: {stats['dihedrals']['mean']:.1f} ± {stats['dihedrals']['std']:.1f}°")
            print(f"    Range: [{stats['dihedrals']['min']:.1f}, {stats['dihedrals']['max']:.1f}]")
    
    if all_stats:
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        
        # Aggregate across structures
        all_dist_means = [s['distances']['mean'] for s in all_stats]
        all_dist_stds = [s['distances']['std'] for s in all_stats]
        all_angle_stds = [s['angles']['std'] for s in all_stats]
        all_dihedral_stds = [s['dihedrals']['std'] for s in all_stats]
        
        print(f"\nCA-CA Distance Statistics:")
        print(f"  Mean across structures: {np.mean(all_dist_means):.3f} ± {np.std(all_dist_means):.3f} Å")
        print(f"  Avg std within structures: {np.mean(all_dist_stds):.3f} Å")
        
        print(f"\nAngle Diversity:")
        print(f"  Avg std within structures: {np.mean(all_angle_stds):.1f}°")
        
        print(f"\nDihedral Diversity:")
        print(f"  Avg std within structures: {np.mean(all_dihedral_stds):.1f}°")
        
        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        
        avg_dist = np.mean(all_dist_means)
        avg_angle_std = np.mean(all_angle_stds)
        avg_dihedral_std = np.mean(all_dihedral_stds)
        
        issues = []
        
        if avg_dist < 3.0:
            issues.append(f"⚠ CA-CA distances too short ({avg_dist:.3f} Å, expected ~3.8 Å)")
        elif avg_dist > 4.5:
            issues.append(f"⚠ CA-CA distances too long ({avg_dist:.3f} Å, expected ~3.8 Å)")
        else:
            print(f"✓ CA-CA distances look good ({avg_dist:.3f} Å)")
        
        if avg_angle_std < 5.0:
            issues.append(f"⚠ Angles too uniform (std={avg_angle_std:.1f}°, expected >10°)")
        else:
            print(f"✓ Angle diversity looks good (std={avg_angle_std:.1f}°)")
        
        if avg_dihedral_std < 20.0:
            issues.append(f"⚠ Dihedrals too uniform (std={avg_dihedral_std:.1f}°, expected >40°)")
        else:
            print(f"✓ Dihedral diversity looks good (std={avg_dihedral_std:.1f}°)")
        
        if issues:
            print("\nISSUES DETECTED:")
            for issue in issues:
                print(f"  {issue}")
            print("\nThis explains why ProteinMPNN generates repetitive sequences!")
            print("The backbone structures are still too uniform/unrealistic.")
        else:
            print("\n✓ Backbone structures look diverse and realistic!")
            print("If MPNN still generates repetitive sequences, the issue is likely:")
            print("  - MPNN temperature too low (try 0.3-0.5)")
            print("  - Need more sequence diversity per structure")

if __name__ == "__main__":
    main()
