#!/bin/bash
# Complete pipeline: Training → Sampling → ProteinMPNN → OmegaFold → Validation

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE FLOW SCAFFOLDING PIPELINE"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Train enhanced flow matching model"
echo "  2. Generate backbone structures"
echo "  3. Design sequences with ProteinMPNN"
echo "  4. Predict full structures with OmegaFold"
echo "  5. Validate and analyze results"
echo ""
echo "================================================================================"
echo ""

# Configuration
EXPERIMENT_NAME="flow_scaf_$(date +%y%m%d_%H%M%S)"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# ============================================================================
# STEP 1: Training
# ============================================================================
echo "================================================================================"
echo "STEP 1: TRAINING ENHANCED FLOW MATCHING MODEL"
echo "================================================================================"
echo ""

if [ -f "results/enhanced_flow/enhanced_flow_*/models/best_by_valid/*.ckpt" ]; then
    echo "Found existing trained model. Skip training? (y/n)"
    read -r skip_training
    if [ "$skip_training" = "y" ]; then
        echo "Skipping training, using existing model..."
        MODEL_DIR=$(ls -td results/enhanced_flow/enhanced_flow_* | head -1)
    else
        echo "Training new model..."
        bash train_and_evaluate_enhanced_flow.sh
        MODEL_DIR=$(ls -td results/enhanced_flow/enhanced_flow_* | head -1)
    fi
else
    echo "No existing model found. Training..."
    bash train_and_evaluate_enhanced_flow.sh
    MODEL_DIR=$(ls -td results/enhanced_flow/enhanced_flow_* | head -1)
fi

echo ""
echo "✓ Model ready: ${MODEL_DIR}"
echo ""

# ============================================================================
# STEP 2: Generate Backbones
# ============================================================================
echo "================================================================================"
echo "STEP 2: GENERATING BACKBONE STRUCTURES"
echo "================================================================================"
echo ""

SAMPLES_DIR="${RESULTS_DIR}/samples"

# Generate different scenarios
# Format: name:length:motifs:guidance
# Leave motifs and guidance empty for unconditional generation
scenarios=(
    "unconditional:100::"
    "single_motif_short:80:30-40:2.0"
    "single_motif_medium:100:30-50:2.0"
    "two_motifs:120:20-30,80-90:2.5"
)

for scenario in "${scenarios[@]}"; do
    IFS=':' read -r name length motifs guidance <<< "$scenario"
    
    echo "Generating ${name} (length=${length}, motifs=${motifs})..."
    
    # Build command with conditional parameters
    cmd="python bin/sample_enhanced_flow.py \
        --model_dir ${MODEL_DIR} \
        --length ${length} \
        --n_samples 20 \
        --num_steps 50 \
        --output_dir ${SAMPLES_DIR}/${name} \
        --save_pdb \
        --save_angles"
    
    # Add motif parameters only if motifs are specified
    if [ -n "${motifs}" ]; then
        cmd="${cmd} --motif_regions ${motifs}"
    fi
    
    if [ -n "${guidance}" ]; then
        cmd="${cmd} --guidance_scale ${guidance}"
    fi
    
    # Execute command
    eval ${cmd}
    
    echo "✓ Generated ${name}"
    echo ""
done

echo "✓ All backbone structures generated"
echo ""

# ============================================================================
# STEP 3: Design Sequences with ProteinMPNN
# ============================================================================
echo "================================================================================"
echo "STEP 3: DESIGNING SEQUENCES WITH PROTEINMPNN"
echo "================================================================================"
echo ""

# Check if PyTorch is available (required for ProteinMPNN)
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠ PyTorch not found - ProteinMPNN requires PyTorch"
    echo ""
    echo "To install PyTorch:"
    echo "  pip install torch"
    echo ""
    echo "Skipping ProteinMPNN sequence design..."
    echo "You can run it manually later with:"
    echo "  python bin/design_sequences_mpnn.py --input_dir <pdb_dir> --output_dir <output_dir>"
    echo ""
    SKIP_MPNN=true
else
    SKIP_MPNN=false
    
    MPNN_DIR="${RESULTS_DIR}/proteinmpnn"
    mkdir -p "${MPNN_DIR}"

    for scenario_dir in "${SAMPLES_DIR}"/*; do
        scenario_name=$(basename "${scenario_dir}")
        pdb_dir="${scenario_dir}/pdb"
        
        if [ ! -d "${pdb_dir}" ]; then
            echo "⚠ No PDB directory found for ${scenario_name}, skipping..."
            continue
        fi
        
        echo "Designing sequences for ${scenario_name}..."
        
        python bin/design_sequences_mpnn.py \
            --input_dir "${pdb_dir}" \
            --output_dir "${MPNN_DIR}/${scenario_name}" \
            --num_sequences 3 \
            --temperature 0.1 \
            --max_structures 5
        
        echo "✓ Sequences designed for ${scenario_name}"
        echo ""
    done

    echo "✓ All sequences designed"
    echo ""
fi

# ============================================================================
# STEP 4: Predict Structures with OmegaFold
# ============================================================================
echo "================================================================================"
echo "STEP 4: PREDICTING FULL STRUCTURES WITH OMEGAFOLD"
echo "================================================================================"
echo ""

OMEGAFOLD_DIR="${RESULTS_DIR}/omegafold"
mkdir -p "${OMEGAFOLD_DIR}"

# Check if ProteinMPNN was skipped
if [ "$SKIP_MPNN" = true ]; then
    echo "⚠ Skipping OmegaFold (ProteinMPNN was skipped)"
    SKIP_OMEGAFOLD=true
# Check if OmegaFold conda environment is activated
elif [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "omegafold" ]; then
    echo "⚠ OmegaFold conda environment not activated"
    echo ""
    echo "To use OmegaFold, activate the environment first:"
    echo "  conda activate omegafold"
    echo "  bash run_complete_pipeline.sh"
    echo ""
    echo "If not installed:"
    echo "  conda create -n omegafold python=3.9"
    echo "  conda activate omegafold"
    echo "  pip install omegafold"
    echo ""
    echo "Skipping OmegaFold prediction..."
    SKIP_OMEGAFOLD=true
elif ! command -v omegafold &> /dev/null; then
    echo "⚠ OmegaFold command not found"
    echo ""
    echo "Install OmegaFold in the current environment:"
    echo "  pip install omegafold"
    echo ""
    echo "Skipping OmegaFold prediction..."
    SKIP_OMEGAFOLD=true
else
    echo "✓ OmegaFold environment activated: $CONDA_DEFAULT_ENV"
    SKIP_OMEGAFOLD=false
    
    for scenario_dir in "${MPNN_DIR}"/*; do
        scenario_name=$(basename "${scenario_dir}")
        
        # Find all FASTA files
        fasta_files=("${scenario_dir}"/*.fa)
        
        if [ ! -f "${fasta_files[0]}" ]; then
            echo "⚠ No FASTA files found for ${scenario_name}, skipping..."
            continue
        fi
        
        echo "Predicting structures for ${scenario_name}..."
        
        # Run OmegaFold on all sequences
        python bin/omegafold_across_gpus.py \
            "${scenario_dir}"/*.fa \
            --outdir "${OMEGAFOLD_DIR}/${scenario_name}" \
            --gpus 0
        
        echo "✓ Structures predicted for ${scenario_name}"
        echo ""
    done
    
    echo "✓ All structures predicted"
    echo ""
fi

# ============================================================================
# STEP 5: Validation and Analysis
# ============================================================================
echo "================================================================================"
echo "STEP 5: VALIDATION AND ANALYSIS"
echo "================================================================================"
echo ""

VALIDATION_DIR="${RESULTS_DIR}/validation"
mkdir -p "${VALIDATION_DIR}"

if [ "$SKIP_OMEGAFOLD" = false ]; then
    for scenario_dir in "${SAMPLES_DIR}"/*; do
        scenario_name=$(basename "${scenario_dir}")
        
        echo "Validating ${scenario_name}..."
        
        python bin/validate_structures.py \
            --backbone_dir "${scenario_dir}/pdb" \
            --sequences_dir "${MPNN_DIR}/${scenario_name}" \
            --predicted_dir "${OMEGAFOLD_DIR}/${scenario_name}" \
            --output_dir "${VALIDATION_DIR}/${scenario_name}"
        
        echo "✓ Validation complete for ${scenario_name}"
        echo ""
    done
    
    echo "✓ All validations complete"
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Directory structure:"
echo "  ${RESULTS_DIR}/"
echo "  ├── samples/              # Generated backbones"
echo "  ├── proteinmpnn/          # Designed sequences"
if [ "$SKIP_OMEGAFOLD" = false ]; then
echo "  ├── omegafold/            # Predicted structures"
echo "  └── validation/           # Validation reports"
else
echo "  └── (omegafold skipped - not installed)"
fi
echo ""
echo "Next steps:"
echo "  1. Review validation reports in ${VALIDATION_DIR}"
echo "  2. Analyze TM-scores and RMSD values"
echo "  3. Select best designs for experimental validation"
echo ""
echo "================================================================================"
