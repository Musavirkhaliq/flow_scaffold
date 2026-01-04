#!/usr/bin/env python3
"""
Helper script to apply loss weight configuration to enhanced_models_v2.py
"""

import argparse
import re
from pathlib import Path

def apply_loss_weights(omega_weight, rama_penalty, geometric_weight=None):
    """Apply loss weights to enhanced_models_v2.py"""
    
    file_path = Path('foldingdiff/enhanced_models_v2.py')
    content = file_path.read_text()
    
    # Update omega weight (line ~1347)
    # Pattern: omega_penalty = omega_penalty * 1.0
    old_omega = re.search(r'omega_penalty = omega_penalty \* (\d+\.\d+)', content)
    if old_omega:
        content = re.sub(
            r'omega_penalty = omega_penalty \* \d+\.\d+',
            f'omega_penalty = omega_penalty * {omega_weight}',
            content
        )
        print(f"✓ Updated omega weight: {old_omega.group(1)} → {omega_weight}")
    else:
        print("⚠️  Could not find omega weight pattern")
    
    # Update Ramachandran penalty (line ~1333)
    # Pattern: rama_loss = outlier_fraction * 2.0 - favored_fraction * 0.3
    old_rama = re.search(r'rama_loss = outlier_fraction \* (\d+\.\d+) - favored_fraction \* (\d+\.\d+)', content)
    if old_rama:
        content = re.sub(
            r'rama_loss = outlier_fraction \* \d+\.\d+ - favored_fraction \* \d+\.\d+',
            f'rama_loss = outlier_fraction * {rama_penalty} - favored_fraction * 0.3',
            content
        )
        print(f"✓ Updated Ramachandran penalty: {old_rama.group(1)} → {rama_penalty}")
    else:
        print("⚠️  Could not find Ramachandran penalty pattern")
    
    # Write back
    file_path.write_text(content)
    print(f"✓ Applied loss weights to {file_path}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply loss weights to model")
    parser.add_argument("--omega_weight", type=float, required=True)
    parser.add_argument("--rama_penalty", type=float, required=True)
    parser.add_argument("--geometric_weight", type=float, default=None,
                       help="Geometric weight (set in training script, not code)")
    
    args = parser.parse_args()
    
    apply_loss_weights(args.omega_weight, args.rama_penalty, args.geometric_weight)
    
    if args.geometric_weight:
        print(f"\n⚠️  Note: Geometric weight ({args.geometric_weight}) should be set in training script")
        print("   Edit train_and_evaluate_advanced_flow.sh: GEOMETRIC_WEIGHT=...")



