# PLM_MODEL Setup and Verification

**Date:** 2026-01-03  
**Status:** ✅ **READY TO USE**

---

## Overview

The Protein Language Model (PLM) `facebook/esm2_t12_35M_UR50D` is used for sequence-augmented flow matching, following the FoldFlow++ approach.

---

## Verification Results

### ✅ Dependencies Installed

- **transformers:** 4.48.2 ✅
- **torch:** 2.6.0+cu124 ✅
- **CUDA:** Available (4 GPUs) ✅

### ✅ Model Status

- **Tokenizer:** Loaded successfully ✅
- **Model:** Loaded successfully ✅
- **Cache:** `/home/masavir/.cache/huggingface/hub` ✅

---

## Model Details

- **Model Name:** `facebook/esm2_t12_35M_UR50D`
- **Model Size:** ~500MB
- **Purpose:** Sequence embeddings for flow matching
- **Usage:** Feature extraction (frozen weights)

---

## How It's Used

### 1. During Training

The PLM is loaded in `foldingdiff/advanced_flow_matching.py`:

```python
from transformers import EsmModel, EsmTokenizer

self.tokenizer = EsmTokenizer.from_pretrained(plm_model)
self.plm = EsmModel.from_pretrained(plm_model)

# Freeze PLM weights (feature extraction only)
for param in self.plm.parameters():
    param.requires_grad = False
```

### 2. Sequence Encoding

Sequences are encoded using the PLM:

```python
def encode_sequence(self, sequences: List[str], device: torch.device):
    inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = self.plm(**inputs)
    return outputs.last_hidden_state
```

### 3. Multi-Modal Fusion

Sequence embeddings are fused with structural features using cross-attention:

```python
# In BertForAdvancedFlowMatching
if fusion_mode == "cross_attn":
    self.fusion_layer = CrossModalAttention(config.hidden_size)
```

---

## Configuration

### Shell Script (`train_and_evaluate_advanced_flow.sh`)

```bash
PLM_MODEL="facebook/esm2_t12_35M_UR50D"  # Protein language model
FUSION_MODE="cross_attn"  # Multi-modal fusion strategy
```

### Training Script (`bin/train_advanced_flow.py`)

```python
parser.add_argument("--plm_model", type=str, default="facebook/esm2_t12_35M_UR50D")
parser.add_argument("--fusion_mode", type=str, default="cross_attn")
```

### Model Creation

```python
model = BertForAdvancedFlowMatchingTraining(
    ...
    plm_model=args.plm_model,
    fusion_mode=args.fusion_mode,
    ...
)
```

---

## Automatic Download

The model is automatically downloaded from HuggingFace on first use if not cached.

**Cache Location:** `~/.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D`

---

## Troubleshooting

### If Model Fails to Load

1. **Check transformers installation:**
   ```bash
   pip install transformers
   ```

2. **Check internet connection** (for first download)

3. **Check disk space** (~500MB required)

4. **Manual download:**
   ```python
   from transformers import EsmModel
   EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
   ```

### Warning About Uninitialized Weights

The warning:
```
Some weights of EsmModel were not initialized from the model checkpoint
```

This is **normal** and expected. ESM models are used for feature extraction, not fine-tuning. The uninitialized weights (pooler layer) are not used.

---

## Expected Behavior

During training, you should see:
```
Loaded protein language model: facebook/esm2_t12_35M_UR50D
SequenceAugmentedFlowMatching: PLM=True, fusion=cross_attn
✓ Sequence augmentation with facebook/esm2_t12_35M_UR50D
✓ Multi-modal fusion: cross_attn
```

---

## Summary

✅ **PLM_MODEL is fully configured and ready to use**

- Model is installed and accessible
- Will be automatically used during training
- Provides sequence embeddings for better flow matching
- Matches best run configuration

**No action required** - the setup is complete!



