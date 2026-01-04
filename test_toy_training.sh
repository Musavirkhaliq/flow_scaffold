#!/bin/bash
# Quick test with toy dataset (50 train + 10 validation samples)
# Usage: ./test_toy_training.sh

set -e

echo "=========================================="
echo "TOY DATASET TESTING"
echo "Quick test with 50 train + 10 validation samples"
echo "=========================================="
echo ""

EXPERIMENT_NAME="test_toy_$(date +%y%m%d_%H%M%S)"
OUTPUT_DIR="results/advanced_flow"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo ""

# Quick test parameters
EPOCHS=2
BATCH_SIZE=4
HIDDEN_SIZE=256
NUM_LAYERS=4
NUM_HEADS=8
PAD=128

echo "Test parameters:"
echo "  - Epochs: ${EPOCHS} (quick test)"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Hidden size: ${HIDDEN_SIZE} (reduced for speed)"
echo "  - Layers: ${NUM_LAYERS} (reduced for speed)"
echo "  - Heads: ${NUM_HEADS} (reduced for speed)"
echo "  - Pad: ${PAD} (reduced for speed)"
echo "  - Dataset: Toy (50 train + 10 validation)"
echo ""

# Train with toy dataset
echo "Starting training..."
python bin/train_advanced_flow.py \
    --toy \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accumulate_grad_batches 1 \
    --lr 1e-4 \
    --lr_scheduler CosineAnnealing \
    --warmup_ratio 0.1 \
    --gradient_clip 1.0 \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --pad ${PAD} \
    --min_length 40 \
    --motif_length_min 5 \
    --motif_length_max 20 \
    --motif_prob 0.8 \
    --guidance_dropout 0.15 \
    --timesteps 1000 \
    --beta_schedule cosine \
    --use_coords \
    --use_local_frames \
    --use_pairwise \
    --use_sequence \
    --use_ss \
    --use_sequence_augmentation \
    --use_geometric_inverse_design \
    --use_multiscale_attention \
    --use_multiscale_loss \
    --use_consistency_loss \
    --use_geometric_loss \
    --consistency_weight 0.1 \
    --geometric_weight 0.22 \
    --plm_model facebook/esm2_t12_35M_UR50D \
    --fusion_mode cross_attn \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --gpus 1 \
    --num_workers 2 \
    --seed 42

echo ""
echo "✓ Toy dataset training complete!"
echo "Model saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo ""
echo "To test sampling, run:"
echo "  python bin/sample_advanced_flow.py \\"
echo "      --model_dir ${OUTPUT_DIR}/${EXPERIMENT_NAME} \\"
echo "      --length 50 \\"
echo "      --n_samples 3 \\"
echo "      --num_steps 20 \\"
echo "      --device cuda:0"
echo ""


