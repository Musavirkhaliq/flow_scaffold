# Standalone Scripts Guide

This guide covers the standalone scripts for running ProteinMPNN and OmegaFold independently.

## Overview

The standalone scripts allow you to run each step of the pipeline independently:

1. **run_proteinmpnn.sh** - Design sequences for backbone structures
2. **run_omegafold.sh** - Predict full structures from sequences

## Prerequisites

### For ProteinMPNN
```bash
# Install PyTorch
pip install torch

# Clone ProteinMPNN
git clone https://github.com/dauparas/ProteinMPNN.git
```

### For OmegaFold
```bash
# Create conda environment
conda create -n omegafold python=3.9
conda activate omegafold

# Install OmegaFold
pip install omegafold
```

## ProteinMPNN - Sequence Design

### Basic Usage

```bash
bash run_proteinmpnn.sh --input_dir <pdb_directory>
```

### Full Options

```bash
bash run_proteinmpnn.sh \
  --input_dir samples/test/pdb \
  --output_dir sequences/test \
  --num_sequences 5 \
  --temperature 0.1 \
  --max_structures 10
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Directory with PDB files (required) | - |
| `--output_dir` | Output directory | `sequences` |
| `--num_sequences` | Sequences per structure | `3` |
| `--temperature` | Sampling temperature (0.1-1.0) | `0.1` |
| `--max_structures` | Max structures to process | all |

### Temperature Guide

- **0.1** - Conservative, similar sequences
- **0.3** - Moderate diversity
- **0.5** - High diversity
- **1.0** - Maximum diversity

### Example Workflow

```bash
# 1. Generate backbones
python bin/sample_enhanced_flow.py \
  --model_dir results/enhanced_flow/enhanced_flow_* \
  --length 100 --n_samples 20 \
  --output_dir samples/my_design --save_pdb

# 2. Design sequences
bash run_proteinmpnn.sh \
  --input_dir samples/my_design/pdb \
  --output_dir sequences/my_design \
  --num_sequences 5 \
  --temperature 0.2

# Output: sequences/my_design/*.fa
```

### Output Structure

```
sequences/my_design/
├── sample_0000/
│   ├── seq_0.fa
│   ├── seq_1.fa
│   └── seq_2.fa
├── sample_0001/
│   ├── seq_0.fa
│   ├── seq_1.fa
│   └── seq_2.fa
└── ...
```

### Troubleshooting

**Error: PyTorch not found**
```bash
pip install torch
```

**Error: ProteinMPNN not found**
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
```

**Error: No PDB files found**
- Check that input directory contains `.pdb` files
- Verify the path is correct

## OmegaFold - Structure Prediction

### Basic Usage

```bash
# Activate OmegaFold environment first!
conda activate omegafold

bash run_omegafold.sh --input_dir <fasta_directory>
```

### Full Options

```bash
conda activate omegafold

bash run_omegafold.sh \
  --input_dir sequences/test \
  --output_dir predictions/test \
  --gpus "0 1 2 3" \
  --batch_size 1
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Directory with FASTA files (required) | - |
| `--output_dir` | Output directory | `predictions` |
| `--gpus` | GPU IDs (space-separated) | `0` |
| `--batch_size` | Batch size | `1` |

### GPU Configuration

**Single GPU:**
```bash
bash run_omegafold.sh --input_dir sequences --gpus "0"
```

**Multiple GPUs (parallel):**
```bash
bash run_omegafold.sh --input_dir sequences --gpus "0 1 2 3"
```

This will distribute sequences across GPUs for faster processing.

### Example Workflow

```bash
# 1. Activate OmegaFold environment
conda activate omegafold

# 2. Predict structures
bash run_omegafold.sh \
  --input_dir sequences/my_design \
  --output_dir predictions/my_design \
  --gpus "0 1"

# Output: predictions/my_design/*.pdb
```

### Output Structure

```
predictions/my_design/
├── seq_0.pdb
├── seq_1.pdb
├── seq_2.pdb
└── ...
```

### Performance

| GPUs | Sequences | Time |
|------|-----------|------|
| 1 | 60 | ~60 min |
| 2 | 60 | ~30 min |
| 4 | 60 | ~15 min |

### Troubleshooting

**Error: OmegaFold not found**
```bash
conda create -n omegafold python=3.9
conda activate omegafold
pip install omegafold
```

**Error: CUDA out of memory**
- Use fewer GPUs
- Process fewer sequences at once
- Reduce batch size

**Error: No FASTA files found**
- Check that input directory contains `.fa` or `.fasta` files
- Verify the path is correct

## Complete Example Workflow

### Step-by-Step

```bash
# 1. Generate backbones (already done)
# samples/my_design/pdb/*.pdb

# 2. Design sequences with ProteinMPNN
bash run_proteinmpnn.sh \
  --input_dir samples/my_design/pdb \
  --output_dir sequences/my_design \
  --num_sequences 3 \
  --temperature 0.1

# 3. Predict structures with OmegaFold
conda activate omegafold
bash run_omegafold.sh \
  --input_dir sequences/my_design \
  --output_dir predictions/my_design \
  --gpus "0 1"

# 4. Validate results
python bin/validate_structures.py \
  --backbone_dir samples/my_design/pdb \
  --sequences_dir sequences/my_design \
  --predicted_dir predictions/my_design \
  --output_dir validation/my_design
```

### Batch Processing Multiple Experiments

```bash
# Process multiple sample directories
for sample_dir in samples/*/pdb; do
    name=$(basename $(dirname $sample_dir))
    
    echo "Processing $name..."
    
    # ProteinMPNN
    bash run_proteinmpnn.sh \
      --input_dir "$sample_dir" \
      --output_dir "sequences/$name" \
      --num_sequences 3
    
    # OmegaFold
    conda activate omegafold
    bash run_omegafold.sh \
      --input_dir "sequences/$name" \
      --output_dir "predictions/$name" \
      --gpus "0"
done
```

## Tips and Best Practices

### ProteinMPNN

1. **Start conservative**: Use temperature 0.1 for initial designs
2. **Generate multiple sequences**: 3-10 per backbone for diversity
3. **Batch processing**: Process multiple structures at once
4. **Check outputs**: Verify FASTA files are generated correctly

### OmegaFold

1. **Use multiple GPUs**: Significantly speeds up processing
2. **Monitor memory**: Watch GPU memory usage
3. **Batch similar lengths**: Group sequences by length for efficiency
4. **Check confidence**: Review pLDDT scores in output

### General

1. **Organize outputs**: Use descriptive directory names
2. **Keep backbones**: Save original PDB files for validation
3. **Document parameters**: Record settings used for each run
4. **Validate results**: Always run validation after prediction

## Integration with Main Pipeline

These standalone scripts are also used by the main pipeline:

```bash
# Main pipeline uses these internally
bash run_complete_pipeline.sh
```

But you can run them independently for:
- Testing different parameters
- Reprocessing specific samples
- Debugging issues
- Custom workflows

## Quick Reference

### ProteinMPNN
```bash
bash run_proteinmpnn.sh --input_dir <pdb_dir> [--num_sequences N] [--temperature T]
```

### OmegaFold
```bash
conda activate omegafold
bash run_omegafold.sh --input_dir <fasta_dir> [--gpus "0 1 2 3"]
```

### Help
```bash
bash run_proteinmpnn.sh --help
bash run_omegafold.sh --help
```

## Troubleshooting Summary

| Issue | Solution |
|-------|----------|
| PyTorch not found | `pip install torch` |
| ProteinMPNN not found | `git clone https://github.com/dauparas/ProteinMPNN.git` |
| OmegaFold not found | `conda activate omegafold` or install it |
| CUDA out of memory | Use fewer GPUs or smaller batches |
| No output files | Check input directory and file formats |
| Slow processing | Use multiple GPUs for OmegaFold |

## Support

For issues:
1. Check error messages carefully
2. Verify all dependencies are installed
3. Test with small datasets first
4. Review this guide for common solutions

---
**Version**: 1.0  
**Date**: December 18, 2024
