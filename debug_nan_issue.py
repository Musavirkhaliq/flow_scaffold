#!/usr/bin/env python3
"""
Debug script to isolate the NaN issue in advanced flow matching model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_coordinate_computation():
    """Test if coordinate computation produces NaN"""
    print("Testing coordinate computation...")
    
    from foldingdiff.embeddings import angles_to_coords_simple
    
    # Create sample angles
    batch_size, seq_len = 2, 10
    angles = torch.randn(batch_size, seq_len, 6)
    
    print(f"Input angles shape: {angles.shape}")
    print(f"Input angles range: [{angles.min():.3f}, {angles.max():.3f}]")
    print(f"Input has NaN: {torch.isnan(angles).any()}")
    
    # Compute coordinates
    coords = angles_to_coords_simple(angles)
    
    print(f"Output coords shape: {coords.shape}")
    print(f"Output coords range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"Output has NaN: {torch.isnan(coords).any()}")
    
    return not torch.isnan(coords).any()

def test_embedding_layer():
    """Test if embedding layer produces NaN"""
    print("\nTesting embedding layer...")
    
    from foldingdiff.embeddings import ProteinStructureEmbedding
    
    # Create embedding layer
    config = {
        'use_sequence': True,
        'use_coords': True,
        'use_local_frames': True,
        'use_pairwise': True,
        'use_secondary_structure': True,
    }
    
    embedder = ProteinStructureEmbedding(
        hidden_size=256,
        **config
    )
    
    # Create sample inputs
    batch_size, seq_len = 2, 10
    angles = torch.randn(batch_size, seq_len, 6)
    coords = torch.randn(batch_size, seq_len, 4, 3)
    aa_types = torch.randint(0, 20, (batch_size, seq_len))
    attn_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input angles has NaN: {torch.isnan(angles).any()}")
    print(f"Input coords has NaN: {torch.isnan(coords).any()}")
    
    # Forward pass
    try:
        output = embedder(
            angles=angles,
            coords=coords,
            aa_types=aa_types,
            attn_mask=attn_mask
        )
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Output has NaN: {torch.isnan(output).any()}")
        
        return not torch.isnan(output).any()
        
    except Exception as e:
        print(f"Error in embedding layer: {e}")
        return False

def test_advanced_model_components():
    """Test advanced model components individually"""
    print("\nTesting advanced model components...")
    
    from foldingdiff.advanced_flow_matching import create_advanced_flow_matching_model
    
    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
    )
    
    try:
        components = create_advanced_flow_matching_model(
            config,
            use_sequence_augmentation=True,
            use_motif_amortization=True,
            use_geometric_inverse_design=True,
            use_optimal_transport=False,
            use_multiscale=True,
        )
        
        print(f"Created components: {list(components.keys())}")
        
        # Test sequence augmentation
        if "sequence_augmented" in components:
            seq_aug = components["sequence_augmented"]
            sequences = ["ACDEFGHIKLMNPQRSTVWY"]
            device = torch.device("cpu")
            
            try:
                seq_features = seq_aug.encode_sequence(sequences, device)
                if seq_features is not None:
                    print(f"Sequence features shape: {seq_features.shape}")
                    print(f"Sequence features has NaN: {torch.isnan(seq_features).any()}")
                else:
                    print("Sequence features is None (PLM not available)")
            except Exception as e:
                print(f"Error in sequence augmentation: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error creating advanced components: {e}")
        return False

def test_model_initialization():
    """Test if model initialization produces NaN weights"""
    print("\nTesting model initialization...")
    
    from foldingdiff.enhanced_models_v2 import BertForAdvancedFlowMatchingTraining
    
    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    
    embedding_config = {
        'use_sequence': True,
        'use_coords': True,
        'use_local_frames': True,
        'use_pairwise': True,
        'use_secondary_structure': True,
    }
    
    try:
        model = BertForAdvancedFlowMatchingTraining(
            config=config,
            ft_is_angular=[True, True, True, False, False, False],
            ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            use_enhanced_embedding=True,
            embedding_config=embedding_config,
            use_motif_conditioning=True,
            
            # Advanced features (disable some for debugging)
            use_sequence_augmentation=False,  # Disable to isolate issue
            use_geometric_inverse_design=False,
            use_multiscale_attention=False,
            
            # Training parameters
            lr=1e-4,
            epochs=1,
            steps_per_epoch=10,
        )
        
        # Check for NaN in model weights
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in parameter: {name}")
                has_nan = True
        
        if not has_nan:
            print("No NaN found in model parameters")
        
        return not has_nan
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

def test_simple_forward_pass():
    """Test a simple forward pass"""
    print("\nTesting simple forward pass...")
    
    from foldingdiff.enhanced_models_v2 import BertForAdvancedFlowMatchingTraining
    
    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    
    embedding_config = {
        'use_sequence': False,  # Disable to isolate
        'use_coords': False,
        'use_local_frames': False,
        'use_pairwise': False,
        'use_secondary_structure': False,
    }
    
    try:
        model = BertForAdvancedFlowMatchingTraining(
            config=config,
            ft_is_angular=[True, True, True, False, False, False],
            ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            use_enhanced_embedding=True,
            embedding_config=embedding_config,
            use_motif_conditioning=False,  # Disable
            
            # Disable all advanced features
            use_sequence_augmentation=False,
            use_geometric_inverse_design=False,
            use_multiscale_attention=False,
            
            lr=1e-4,
            epochs=1,
            steps_per_epoch=10,
        )
        
        # Create simple inputs
        batch_size, seq_len = 2, 10
        inputs = torch.randn(batch_size, seq_len, 6)
        timestep = torch.rand(batch_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        print(f"Input shape: {inputs.shape}")
        print(f"Input has NaN: {torch.isnan(inputs).any()}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model.forward(
                inputs=inputs,
                timestep=timestep,
                attention_mask=attention_mask,
            )
        
        print(f"Output shape: {output.shape}")
        print(f"Output has NaN: {torch.isnan(output).any()}")
        
        if torch.isnan(output).any():
            print("NaN detected in simple forward pass!")
            return False
        else:
            print("Simple forward pass successful")
            return True
        
    except Exception as e:
        print(f"Error in simple forward pass: {e}")
        return False

def main():
    """Run all tests to isolate NaN issue"""
    print("=" * 60)
    print("DEBUGGING NaN ISSUE IN ADVANCED FLOW MATCHING")
    print("=" * 60)
    
    tests = [
        ("Coordinate computation", test_coordinate_computation),
        ("Embedding layer", test_embedding_layer),
        ("Advanced components", test_advanced_model_components),
        ("Model initialization", test_model_initialization),
        ("Simple forward pass", test_simple_forward_pass),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<40} {status}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not results.get("Coordinate computation", True):
        print("• Fix coordinate computation in angles_to_coords_simple")
    
    if not results.get("Embedding layer", True):
        print("• Fix embedding layer in ProteinStructureEmbedding")
    
    if not results.get("Advanced components", True):
        print("• Fix advanced flow matching components")
    
    if not results.get("Model initialization", True):
        print("• Fix model weight initialization")
    
    if not results.get("Simple forward pass", True):
        print("• Issue is in basic model architecture")
    
    if all(results.values()):
        print("• All tests passed - issue might be in training loop or data")
    
    print("=" * 60)

if __name__ == "__main__":
    main()