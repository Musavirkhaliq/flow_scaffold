#!/usr/bin/env python3
"""
Test if the advanced training script can start without errors.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    print("Testing imports...")
    
    try:
        from foldingdiff.enhanced_datasets import create_enhanced_dataset
        print("✓ Enhanced datasets import OK")
    except Exception as e:
        print(f"✗ Enhanced datasets import failed: {e}")
        return False
    
    try:
        from foldingdiff.enhanced_models_v2 import BertForAdvancedFlowMatchingTraining
        print("✓ Advanced models import OK")
    except Exception as e:
        print(f"✗ Advanced models import failed: {e}")
        return False
    
    try:
        from transformers import BertConfig
        print("✓ Transformers import OK")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    return True

def test_dataset_creation():
    print("\nTesting dataset creation...")
    
    try:
        from foldingdiff.enhanced_datasets import create_enhanced_dataset
        
        # Create very small dataset for testing
        dataset = create_enhanced_dataset(
            split=None,  # No split for toy
            pad=64,      # Smaller pad
            min_length=20,  # Smaller min length
            toy=2,       # Very small
            compute_coords=True,
            compute_ss=True,
            include_sequences=True,
            use_motif_scaffolding=False,  # Disable for simplicity
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test one item
        item = dataset[0]
        print(f"✓ Item keys: {list(item.keys())}")
        
        if 'sequences' in item:
            print(f"✓ Sequences included: {len(item['sequences'])} sequences")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING ADVANCED TRAINING COMPONENTS")
    print("=" * 50)
    
    if not test_imports():
        print("\n✗ Import tests failed")
        sys.exit(1)
    
    if not test_dataset_creation():
        print("\n✗ Dataset tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("Advanced training should work!")
    print("=" * 50)