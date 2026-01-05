# Toy Dataset Testing Guide
## Quick Testing with Small Dataset

**Purpose:** Test training and sampling quickly with a small toy dataset (50 train + 10 validation samples)

---

## Method 1: Direct Python Command (Recommended for Quick Testing)

### Basic Toy Dataset Training
```bash
python bin/train_advanced_flow.py \
    --toy \
    --epochs 2 \
    --batch_size 4 \
    --output_dir results/advanced_flow \
    --experiment_name test_toy \
    --gpus 1
```

### Full Toy Dataset Training with All Features
```bash
python bin/train_advanced_flow.py \
    --toy \
    --epochs 2 \
    --batch_size 4 \
    --use_coords \
    --use_local_frames \
    --use_pairwise \
    --use_sequence \
    --use_ss \
    --use_sequence_augmentation \
    --use_geometric_inverse_design \
    --use_multiscale_attention \
    --use_geometric_loss \
    --use_oat_fm \
    --plm_model facebook/esm2_t12_35M_UR50D \
    --fusion_mode cross_attn \
    --output_dir results/advanced_flow \
    --experiment_name test_toy_full \
    --gpus 1
```

---

## Method 2: Modify Phase Script

### Option A: Add --toy flag to phase1_train_advanced_flow.sh

Edit `phase1_train_advanced_flow.sh` and add `--toy` to the training command:

```bash
# Around line 95, modify:
TRAIN_CMD="python bin/train_advanced_flow.py \
    --toy \
    --data_dir ${CATH_DIR} \
    ...
```

### Option B: Create a Quick Test Script

Create `test_toy_training.sh`:

```bash
#!/bin/bash
# Quick test with toy dataset

EXPERIMENT_NAME="test_toy_$(date +%y%m%d_%H%M%S)"
OUTPUT_DIR="results/advanced_flow"

python bin/train_advanced_flow.py \
    --toy \
    --epochs 2 \
    --batch_size 4 \
    --accumulate_grad_batches 1 \
    --lr 1e-4 \
    --epochs 2 \
    --lr_scheduler CosineAnnealing \
    --warmup_ratio 0.1 \
    --gradient_clip 1.0 \
    --hidden_size 256 \
    --num_layers 4 \
    --num_heads 8 \
    --pad 128 \
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

echo "✓ Toy dataset training complete!"
echo "Model saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
```

Make it executable:
```bash
chmod +x test_toy_training.sh
./test_toy_training.sh
```

---

## Method 3: Modify Config File (Temporary)

Add a `TOY_MODE` variable to `config_advanced_flow.sh`:

```bash
# Add to config_advanced_flow.sh
TOY_MODE=false  # Set to true for quick testing

# Then modify phase1_train_advanced_flow.sh to use it:
if [ "$TOY_MODE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --toy"
fi
```

---

## Toy Dataset Details

When `--toy` flag is used:
- **Training samples:** 50 structures
- **Validation samples:** 10 structures
- **No train/val split:** Uses all available data
- **Fast iteration:** Perfect for testing code changes

---

## Quick Testing Workflow

### 1. Test Training (2 epochs, ~5-10 minutes)
```bash
python bin/train_advanced_flow.py \
    --toy \
    --epochs 2 \
    --batch_size 4 \
    --output_dir results/advanced_flow \
    --experiment_name test_toy \
    --gpus 1
```

### 2. Test Sampling (after training)
```bash
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/test_toy \
    --length 50 \
    --n_samples 5 \
    --num_steps 20 \
    --device cuda:0
```

### 3. Test Full Pipeline (Training + Sampling)
```bash
# Train
python bin/train_advanced_flow.py --toy --epochs 2 --batch_size 4 \
    --output_dir results/advanced_flow --experiment_name test_toy --gpus 1

# Sample
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/test_toy \
    --length 50 --n_samples 5 --num_steps 20 --device cuda:0
```

---

## Recommended Toy Testing Parameters

For fastest testing:
- **Epochs:** 1-2 (just to verify training runs)
- **Batch size:** 4-8 (smaller for faster iteration)
- **Hidden size:** 256 (smaller model)
- **Num layers:** 4 (fewer layers)
- **Num heads:** 8 (fewer attention heads)
- **Pad:** 128 (smaller padding)
- **Num steps (sampling):** 20 (fewer steps for faster sampling)

---

## Expected Results

With toy dataset:
- **Training time:** ~5-10 minutes (2 epochs)
- **Dataset size:** 50 train + 10 validation
- **Model size:** Smaller (if using reduced hidden_size/layers)
- **Purpose:** Verify code works, not for quality evaluation

---

## Troubleshooting

### Issue: "No data found"
- Make sure CATH dataset is available at `data/cath`
- Or use `--data_dir` to point to your data location

### Issue: "Out of memory"
- Reduce batch size: `--batch_size 2`
- Reduce model size: `--hidden_size 128 --num_layers 2`
- Use CPU: `--gpus 0`

### Issue: "CUDA out of memory"
- Use smaller batch: `--batch_size 2`
- Use CPU: `--gpus 0`
- Reduce model size

---

## Example: Complete Quick Test

```bash
# 1. Train with toy dataset (2 epochs, ~5-10 min)
python bin/train_advanced_flow.py \
    --toy \
    --epochs 2 \
    --batch_size 4 \
    --hidden_size 256 \
    --num_layers 4 \
    --num_heads 8 \
    --pad 128 \
    --use_geometric_loss \
    --output_dir results/advanced_flow \
    --experiment_name quick_test \
    --gpus 1

# 2. Sample a few structures (~1 min)
python bin/sample_advanced_flow.py \
    --model_dir results/advanced_flow/quick_test \
    --length 50 \
    --n_samples 3 \
    --num_steps 20 \
    --device cuda:0

# 3. Check results
ls -lh results/advanced_flow/quick_test/
```

---

**Total time:** ~10-15 minutes for complete test cycle


