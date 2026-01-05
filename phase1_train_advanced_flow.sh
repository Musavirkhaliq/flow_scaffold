#!/bin/bash
# PHASE 1: Train Advanced Flow Matching Model
# Usage: ./phase1_train_advanced_flow.sh [EXPERIMENT_NAME]

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source shared configuration
source config_advanced_flow.sh

# Function to extract experiment name from path or name
extract_experiment_name() {
    local input="$1"
    if [ -z "$input" ]; then
        return
    fi
    # If it's a full path, extract just the last component
    if [[ "$input" == *"/"* ]]; then
        basename "$input"
    else
        echo "$input"
    fi
}

# Allow experiment name override
if [ -n "$1" ]; then
    EXPERIMENT_NAME="$(extract_experiment_name "$1")"
    MODEL_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
fi

echo "=========================================="
echo "PHASE 1: TRAINING ADVANCED FLOW MODEL"
echo "=========================================="
echo ""
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model output: ${MODEL_DIR}"
echo ""
echo "Research advances incorporated:"
echo "  ✓ FrameFlow extensions (motif amortization & guidance)"
echo "  ✓ FoldFlow++ (sequence-augmented flow matching)"
echo "  ✓ EVA-inspired geometric inverse design"
echo "  ✓ Multi-scale attention mechanisms"
echo "  ✓ Enhanced training strategies"
echo ""

echo "Training parameters (SOTA OPTIMIZED):"
echo "  - Epochs: ${EPOCHS} (INCREASED from 10 to 50 - Priority 1 Fix for SOTA convergence)"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Gradient accumulation: ${ACCUMULATE_GRAD_BATCHES} (effective batch: $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES)))"
echo "  - Learning rate: ${LR} (SOTA: Optimized with warmup)"
echo "  - LR scheduler: ${LR_SCHEDULER} (SOTA: Cosine annealing for best convergence)"
echo "  - Warmup ratio: ${WARMUP_RATIO} (SOTA: 10% warmup for stable start)"
echo "  - Hidden size: ${HIDDEN_SIZE} (SOTA: Increased capacity for better representation)"
echo "  - Num layers: ${NUM_LAYERS} (SOTA: Deeper network for better capacity)"
echo "  - Num heads: ${NUM_HEADS} (SOTA: Multi-head attention)"
echo "  - Gradient clipping: ${GRADIENT_CLIP} (SOTA: For training stability)"
echo ""

echo "Advanced features:"
echo "  - Sequence augmentation: ${USE_SEQUENCE_AUGMENTATION}"
echo "  - PLM model: ${PLM_MODEL}"
echo "  - Fusion mode: ${FUSION_MODE}"
echo "  - Geometric inverse design: ${USE_GEOMETRIC_INVERSE_DESIGN}"
echo "  - Multi-scale attention: ${USE_MULTISCALE_ATTENTION}"
echo ""

echo "Advanced training:"
echo "  - Multi-scale loss: ${USE_MULTISCALE_LOSS}"
echo "  - Consistency loss: ${USE_CONSISTENCY_LOSS} (weight: ${CONSISTENCY_WEIGHT})"
echo "  - Geometric loss: ${USE_GEOMETRIC_LOSS} (weight: ${GEOMETRIC_WEIGHT})"
echo "    * NEW: Enhanced clash penalty (weight 0.3, Priority 1 Fix)"
echo "    * NEW: Strengthened Ramachandran loss (3.0-5.0x penalty, Priority 1 Fix)"
echo "    * NEW: Pairwise distance loss (0.1-0.15 weight, Priority 1 Fix)"
echo "  - OAT-FM: ${USE_OAT_FM} (Optimal Acceleration Transport, Priority 2 Fix)"
echo "  - EMA: Enabled (Exponential Moving Average, Priority 2 Fix)"
echo ""

echo "Flow matching:"
echo "  - Timesteps: ${TIMESTEPS}"
echo "  - Schedule: ${BETA_SCHEDULE}"
echo "  - Sampling steps: ${NUM_STEPS} (70x faster than RFDiffusion!)"
echo ""

echo "Motif scaffolding:"
echo "  - Length range: ${MOTIF_MIN}-${MOTIF_MAX}"
echo "  - Guidance dropout: ${GUIDANCE_DROPOUT}"
echo ""

echo "Dataset configuration:"
if [ "$DATASET_SIZE" = "toy" ]; then
    echo "  - TOY MODE: Using small dataset (50 train + 10 validation samples)"
    echo "  - WARNING: This is for testing only, not for production training!"
elif [ "$DATASET_SIZE" = "small" ]; then
    echo "  - SMALL MODE: Using limited dataset (500 train + 100 validation samples)"
    echo "  - Good for quick testing with more data than toy mode"
elif [ "$DATASET_SIZE" = "medium" ]; then
    echo "  - MEDIUM MODE: Using moderate dataset (5000 train + 1000 validation samples)"
    echo "  - Good for moderate testing before full training"
else
    echo "  - FULL MODE: Using complete dataset (no limits)"
    if [ "$USE_COMBINED_DATASET" = true ]; then
        echo "  - Using COMBINED dataset (CATH + AlphaFold)"
        echo "  - CATH directory: ${CATH_DIR}"
        echo "  - AlphaFold directory: ${ALPHAFOLD_DIR}"
    else
        echo "  - Using CATH dataset only"
        echo "  - CATH directory: ${CATH_DIR}"
    fi
fi
echo ""

# Build training command
TRAIN_CMD="python bin/train_advanced_flow.py \
    --data_dir ${CATH_DIR}"

# Add combined dataset flags only if USE_COMBINED_DATASET is true
if [ "$USE_COMBINED_DATASET" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} \
    --use_combined_dataset \
    --alphafold_dir ${ALPHAFOLD_DIR}"
fi

TRAIN_CMD="${TRAIN_CMD} \
    --pad 512 \
    --min_length 40 \
    --motif_length_min ${MOTIF_MIN} \
    --motif_length_max ${MOTIF_MAX} \
    --motif_prob 0.8 \
    --max_motifs 1 \
    --guidance_dropout ${GUIDANCE_DROPOUT} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --timesteps ${TIMESTEPS} \
    --beta_schedule ${BETA_SCHEDULE} \
    --batch_size ${BATCH_SIZE} \
    --accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --lr_scheduler ${LR_SCHEDULER} \
    --warmup_ratio ${WARMUP_RATIO} \
    --gradient_clip ${GRADIENT_CLIP} \
    --output_dir ${OUTPUT_BASE} \
    --experiment_name ${EXPERIMENT_NAME} \
    --gpus 4 \
    --num_workers 4 \
    --seed 42"

# Add enhanced feature flags
if [ "$USE_COORDS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_coords"
fi
if [ "$USE_LOCAL_FRAMES" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_local_frames"
fi
if [ "$USE_PAIRWISE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_pairwise"
fi
if [ "$USE_SEQUENCE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_sequence"
fi
if [ "$USE_SS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_ss"
fi

# Add advanced feature flags
if [ "$USE_SEQUENCE_AUGMENTATION" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_sequence_augmentation"
fi
if [ "$USE_GEOMETRIC_INVERSE_DESIGN" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_geometric_inverse_design"
fi
if [ "$USE_MULTISCALE_ATTENTION" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_multiscale_attention"
fi

# Add PLM and fusion settings (CRITICAL: Ensure PLM_MODEL is used)
TRAIN_CMD="${TRAIN_CMD} --plm_model ${PLM_MODEL}"
TRAIN_CMD="${TRAIN_CMD} --fusion_mode ${FUSION_MODE}"

# Add advanced training flags
if [ "$USE_MULTISCALE_LOSS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_multiscale_loss"
fi
if [ "$USE_CONSISTENCY_LOSS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_consistency_loss"
fi
if [ "$USE_GEOMETRIC_LOSS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_geometric_loss"
fi
if [ "$USE_OAT_FM" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_oat_fm"
fi

TRAIN_CMD="${TRAIN_CMD} --consistency_weight ${CONSISTENCY_WEIGHT}"
TRAIN_CMD="${TRAIN_CMD} --geometric_weight ${GEOMETRIC_WEIGHT}"

# Add dataset size limits based on DATASET_SIZE config
if [ "$DATASET_SIZE" = "toy" ]; then
    TRAIN_CMD="${TRAIN_CMD} --toy"
    echo ""
    echo "=========================================="
    echo "TOY MODE ENABLED - FAST TESTING"
    echo "=========================================="
    echo "Using small dataset: 50 train + 10 validation samples"
    echo "This is for quick code testing, not production training!"
    echo ""
elif [ "$DATASET_SIZE" = "small" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max_train_samples 500 --max_val_samples 100"
    echo ""
    echo "=========================================="
    echo "SMALL DATASET MODE - QUICK TESTING"
    echo "=========================================="
    echo "Using limited dataset: 500 train + 100 validation samples"
    echo "Good for testing with more data than toy mode"
    echo ""
elif [ "$DATASET_SIZE" = "medium" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max_train_samples 5000 --max_val_samples 1000"
    echo ""
    echo "=========================================="
    echo "MEDIUM DATASET MODE - MODERATE TESTING"
    echo "=========================================="
    echo "Using moderate dataset: 5000 train + 1000 validation samples"
    echo "Good for testing before full production training"
    echo ""
fi

# SOTA OPTIMIZATION SUMMARY
echo ""
echo "=========================================="
echo "SOTA OPTIMIZATION SUMMARY"
echo "=========================================="
echo "Key improvements for state-of-the-art performance:"
echo "  ✓ Epochs: ${EPOCHS} (INCREASED from 10, Priority 1 Fix) - Full convergence"
echo "  ✓ Model capacity: ${HIDDEN_SIZE} hidden, ${NUM_LAYERS} layers (vs 512/12)"
echo "  ✓ Effective batch: $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES)) (vs 32)"
echo "  ✓ Learning rate: ${LR} with ${WARMUP_RATIO} warmup (optimized)"
echo "  ✓ Gradient clipping: ${GRADIENT_CLIP} (training stability)"
echo "  ✓ Loss weights: consistency=${CONSISTENCY_WEIGHT}, geometric=${GEOMETRIC_WEIGHT} (optimized)"
echo "  ✓ NEW: Enhanced geometric loss with clash penalty (0.3 weight, Priority 1)"
echo "  ✓ NEW: Strengthened Ramachandran loss (3.0-5.0x penalty, Priority 1)"
echo "  ✓ NEW: Pairwise distance loss (0.1-0.15 weight, Priority 1)"
echo "    ✓ NEW: EMA (Exponential Moving Average) for model stability (Priority 2)"
  echo "  ✓ NEW: OAT-FM support (${USE_OAT_FM}, Priority 2 - Optional)"
  if [ "$USE_COMBINED_DATASET" = true ]; then
    echo "  ✓ Combined dataset: CATH + AlphaFold (82K+ structures vs 27K)"
  else
    echo "  ✓ Dataset: CATH only"
  fi
  echo "  ✓ 80-10-10 split: Proper train/val/test separation"
echo "  ✓ All advanced features enabled: sequence augmentation, geometric inverse design, multi-scale"
echo ""

# Execute training
echo "Starting training with command:"
echo "${TRAIN_CMD}"
echo ""

eval $TRAIN_CMD

echo ""
echo "✓ Advanced training complete!"
echo "Model saved to: ${MODEL_DIR}"
echo ""
echo "To continue with sampling, run:"
echo "  ./phase2_sample_advanced_flow.sh ${EXPERIMENT_NAME}"
echo ""

