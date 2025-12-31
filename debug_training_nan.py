#!/usr/bin/env python3
"""
Debug script to test the actual training scenario that produces NaN.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_dataset_loading():
    """Test if dataset produces valid data"""
    print("Testing dataset loading...")
    
    from foldingdiff.enhanced_datasets import create_enhanced_dataset
    
    try:
        # Create small toy dataset
        dataset = create_enhanced_dataset(
            split=None,  # Use toy
            pad=64,
            min_length=20,
            toy=5,  # Very small
            compute_coords=True,
            compute_ss=True,
            include_sequences=True,
            use_motif_scaffolding=True,
            motif_prob=0.8,
            timesteps=100,
        )
        
        print(f"Dataset created: {len(dataset)} examples")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"\nSample {i}:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        has_nan = torch.isnan(value).any()
                        print(f"  {key}: {value.shape}, NaN: {has_nan}")
                        if has_nan:
                            print(f"    NaN locations: {torch.isnan(value).sum()} / {value.numel()}")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False

def test_dataloader():
    """Test DataLoader with batching"""
    print("\nTesting DataLoader...")
    
    from foldingdiff.enhanced_datasets import create_enhanced_dataset
    from torch.utils.data import DataLoader
    
    try:
        dataset = create_enhanced_dataset(
            split=None,
            pad=64,
            min_length=20,
            toy=5,
            compute_coords=True,
            compute_ss=True,
            include_sequences=True,
            use_motif_scaffolding=True,
            motif_prob=0.8,
            timesteps=100,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # No multiprocessing for debugging
        )
        
        print(f"DataLoader created with {len(dataloader)} batches")
        
        # Test first batch
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    has_nan = torch.isnan(value).any()
                    print(f"  {key}: {value.shape}, NaN: {has_nan}")
                    if has_nan:
                        print(f"    NaN locations: {torch.isnan(value).sum()} / {value.numel()}")
                        print(f"    Sample values: {value.flatten()[:10]}")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
            
            if batch_idx >= 1:  # Only test first 2 batches
                break
        
        return True
        
    except Exception as e:
        print(f"Error with DataLoader: {e}")
        return False

def test_training_step_simulation():
    """Simulate the actual training step that produces NaN"""
    print("\nTesting training step simulation...")
    
    from foldingdiff.enhanced_models_v2 import BertForAdvancedFlowMatchingTraining
    from foldingdiff.enhanced_datasets import create_enhanced_dataset
    from torch.utils.data import DataLoader
    
    try:
        # Create model with advanced features enabled (like in training)
        config = BertConfig(
            hidden_size=256,  # Smaller for debugging
            num_hidden_layers=6,
            num_attention_heads=8,
            max_position_embeddings=64,
        )
        
        embedding_config = {
            'use_sequence': True,
            'use_coords': True,
            'use_local_frames': True,
            'use_pairwise': True,
            'use_secondary_structure': True,
        }
        
        model = BertForAdvancedFlowMatchingTraining(
            config=config,
            ft_is_angular=[True, True, True, False, False, False],
            ft_names=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            use_enhanced_embedding=True,
            embedding_config=embedding_config,
            use_motif_conditioning=True,
            
            # Enable advanced features like in actual training
            use_sequence_augmentation=True,
            use_geometric_inverse_design=True,
            use_multiscale_attention=True,
            plm_model="facebook/esm2_t12_35M_UR50D",
            fusion_mode="cross_attn",
            
            # Training parameters
            lr=1e-4,
            epochs=1,
            steps_per_epoch=10,
        )
        
        # Create dataset
        dataset = create_enhanced_dataset(
            split=None,
            pad=64,
            min_length=20,
            toy=3,  # Very small
            compute_coords=True,
            compute_ss=True,
            include_sequences=True,
            use_motif_scaffolding=True,
            motif_prob=0.8,
            timesteps=100,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        # Test training step
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nTesting training step with batch {batch_idx}")
            
            # Check batch for NaN
            print("Batch contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    has_nan = torch.isnan(value).any()
                    print(f"  {key}: {value.shape}, NaN: {has_nan}")
                    if has_nan:
                        print(f"    First NaN values: {value[torch.isnan(value)][:5]}")
                        return False
            
            # Run training step
            try:
                loss = model.training_step(batch, batch_idx)
                print(f"Training step completed, loss: {loss}")
                
                if torch.isnan(loss):
                    print("NaN loss detected!")
                    return False
                else:
                    print("Training step successful")
                
            except Exception as e:
                print(f"Error in training step: {e}")
                return False
            
            break  # Only test first batch
        
        return True
        
    except Exception as e:
        print(f"Error in training step simulation: {e}")
        return False

def test_sequence_augmentation_issue():
    """Test if sequence augmentation is causing the issue"""
    print("\nTesting sequence augmentation specifically...")
    
    from foldingdiff.advanced_flow_matching import SequenceAugmentedFlowMatching
    
    try:
        # Test with PLM disabled (transformers not available)
        seq_aug = SequenceAugmentedFlowMatching(
            use_plm=True,  # This will fail and disable PLM
            plm_model="facebook/esm2_t12_35M_UR50D",
            fusion_mode="cross_attn"
        )
        
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPKALFWAKRHLPGKPLTFVWKAKGHNPVRAGAATAVAAGTGTSLHHLIAHQVADMFSRMVKQLARHRHGWLKPKMWKGPRLGVPDRHIQQLQKDKEACPLPHAL"]
        device = torch.device("cpu")
        
        seq_features = seq_aug.encode_sequence(sequences, device)
        
        if seq_features is not None:
            print(f"Sequence features shape: {seq_features.shape}")
            print(f"Sequence features has NaN: {torch.isnan(seq_features).any()}")
            
            if torch.isnan(seq_features).any():
                print("NaN found in sequence features!")
                return False
        else:
            print("Sequence features is None (expected - PLM not available)")
        
        return True
        
    except Exception as e:
        print(f"Error in sequence augmentation test: {e}")
        return False

def test_multiscale_attention_issue():
    """Test if multi-scale attention is causing NaN"""
    print("\nTesting multi-scale attention...")
    
    from foldingdiff.enhanced_models_v2 import MultiScaleAttention
    
    try:
        multiscale = MultiScaleAttention(
            hidden_size=256,
            num_scales=3,
            scale_factors=[1, 2, 4]
        )
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 20, 256
        features = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        print(f"Input features shape: {features.shape}")
        print(f"Input has NaN: {torch.isnan(features).any()}")
        
        # Forward pass
        output = multiscale(features, attention_mask)
        
        print(f"Output shape: {output.shape}")
        print(f"Output has NaN: {torch.isnan(output).any()}")
        
        if torch.isnan(output).any():
            print("NaN found in multi-scale attention!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in multi-scale attention test: {e}")
        return False

def main():
    """Run targeted tests for training scenario"""
    print("=" * 60)
    print("DEBUGGING TRAINING NaN ISSUE")
    print("=" * 60)
    
    tests = [
        ("Dataset loading", test_dataset_loading),
        ("DataLoader batching", test_dataloader),
        ("Sequence augmentation", test_sequence_augmentation_issue),
        ("Multi-scale attention", test_multiscale_attention_issue),
        ("Training step simulation", test_training_step_simulation),
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
    
    # Find the issue
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    failed_tests = [name for name, passed in results.items() if not passed]
    
    if not failed_tests:
        print("• All tests passed - NaN issue might be intermittent or in specific conditions")
        print("• Try running actual training with more debugging")
    else:
        print(f"• Failed tests: {', '.join(failed_tests)}")
        print("• Focus on fixing these components first")
    
    print("=" * 60)

if __name__ == "__main__":
    main()