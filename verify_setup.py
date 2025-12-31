#!/usr/bin/env python3
"""Verify that flow_scaf is set up correctly"""
import sys
import os
from pathlib import Path

# Change to script directory
os.chdir(Path(__file__).parent)

print("=" * 80)
print("FLOW_SCAF SETUP VERIFICATION")
print("=" * 80)
print()

# Check imports
print("Checking imports...")
try:
    from foldingdiff.enhanced_datasets import create_enhanced_dataset
    print("  ✓ enhanced_datasets")
except ImportError as e:
    print(f"  ✗ enhanced_datasets: {e}")
    sys.exit(1)

try:
    from foldingdiff.enhanced_models import BertForFlowMatchingEnhanced
    print("  ✓ enhanced_models")
except ImportError as e:
    print(f"  ✗ enhanced_models: {e}")
    sys.exit(1)

try:
    from foldingdiff.flow_matching import compute_angular_flow_matching_loss
    print("  ✓ flow_matching")
except ImportError as e:
    print(f"  ✗ flow_matching: {e}")
    sys.exit(1)

try:
    from foldingdiff.flow_sampling import sample_flow_matching_with_guidance
    print("  ✓ flow_sampling")
except ImportError as e:
    print(f"  ✗ flow_sampling: {e}")
    sys.exit(1)

try:
    from foldingdiff import angles_and_coords, nerf, utils
    print("  ✓ angles_and_coords, nerf, utils")
except ImportError as e:
    print(f"  ✗ core modules: {e}")
    sys.exit(1)

print()

# Check files
print("Checking files...")
required_files = [
    "bin/train_enhanced_flow.py",
    "bin/sample_enhanced_flow.py",
    "bin/convert_angles_to_pdb.py",
    "train_and_evaluate_enhanced_flow.sh",
    "README.md",
    "QUICKSTART.md",
]

for file in required_files:
    if Path(file).exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING")
        sys.exit(1)

print()

# Check data directory
print("Checking data directory...")
data_dir = Path("../data/cath")
if data_dir.exists():
    pdb_files = list(data_dir.glob("dompdb/*"))
    if pdb_files:
        print(f"  ✓ Found {len(pdb_files)} PDB files in {data_dir}")
    else:
        print(f"  ⚠ {data_dir} exists but no PDB files found")
        print(f"    Run: cd ../data && bash download_cath.sh")
else:
    print(f"  ⚠ {data_dir} not found")
    print(f"    Run: cd ../data && bash download_cath.sh")

print()
print("=" * 80)
print("✓✓✓ SETUP VERIFICATION COMPLETE ✓✓✓")
print()
print("Ready to train! Run:")
print("  bash train_and_evaluate_enhanced_flow.sh")
print("=" * 80)
