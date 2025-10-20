#!/bin/bash
# ============================================================================
# EVALUATION SCRIPT FOR SIMILARITY GROUPING METHOD
# ============================================================================
# This script evaluates the CAB-GS method with SIMILARITY GROUPING enabled.
#
# KEY DIFFERENCES from eval_covis_augment_partialscheduler.sh:
#   1. AUGMENTATION: Uses --use_similarity_grouping flag
#      - Matches points with N most co-visible views (not sequential neighbors)
#      - Better correspondence detection
#
#   2. TRAINING: Uses train_similarity.py (similarity_grouping enabled by default)
#      - Dynamic similarity-based view grouping during densification
#      - More adaptive to scene structure
#
# EXPECTED IMPROVEMENTS:
#   - Better point augmentation quality (fewer false positives)
#   - More effective training schedule (adaptive grouping)
#   - Higher metrics (PSNR/SSIM/LPIPS) especially for irregular captures
#
# USAGE:
#   1. Set DATA_ROOT and OUTPUT_ROOT environment variables, or edit paths below
#   2. Run: bash eval_similarity.sh
#
# OUTPUTS:
#   - Augmented point clouds: ${OUTPUT_ROOT}/augmented_similarity/
#   - Trained models: ${OUTPUT_ROOT}/experiments/360_similarity_full/
#   - Metrics: results.json in each scene's model directory
# ============================================================================

# Configuration - UPDATE THESE PATHS FOR YOUR SETUP
n_clusters=4
closest_n_views=12  # Number of most similar views to use in augmentation
# Set these environment variables or edit directly:
# export DATA_ROOT="/path/to/your/data"
# export OUTPUT_ROOT="/path/to/your/output"
colmap_path="${DATA_ROOT:-./data}/mipnerf360"
exp_path="${OUTPUT_ROOT:-./experiments}"
exp_name="360_similarity_full"
augmented_path="${OUTPUT_ROOT:-./output}/augmented_similarity"

# Create augmented path if it doesn't exist
mkdir -p "${augmented_path}"

echo "=========================================="
echo "Experiment: ${exp_name}"
echo "Method: SIMILARITY GROUPING (RECOMMENDED)"
echo "Clusters: ${n_clusters}"
echo "Closest views: ${closest_n_views}"
echo "Dataset: ${colmap_path}"
echo "Augmented output: ${augmented_path}"
echo "Results: ${exp_path}/${exp_name}"
echo "=========================================="

# Process each scene
for scene in $(ls "$colmap_path"); do
    echo ""
    echo "=========================================="
    echo "Processing scene: $scene"
    echo "=========================================="
    
    # Determine image folder based on scene type
    if [ "$scene" = "bicycle" ] || [ "$scene" = "flowers" ] || [ "$scene" = "garden" ] || [ "$scene" = "stump" ] || [ "$scene" = "treehill" ]; then
        images_folder="images_4"
    else
        images_folder="images_2"
    fi
    
    echo "Images folder: ${images_folder}"
    
    # Step 1: Remove old augmented points3D.ply if exists
    rm -f "${augmented_path}/${scene}/sparse/0/points3D.ply"
    
    # Step 2: Augmentation with SIMILARITY-BASED view matching
    echo ""
    echo "Step 1/4: Running augmentation with SIMILARITY GROUPING..."
    python augment.py \
        --colmap_path "${colmap_path}/${scene}/sparse/0" \
        --image_path "${colmap_path}/${scene}/${images_folder}" \
        --augment_path "${augmented_path}/${scene}/sparse/0/points3D.bin" \
        --camera_order covisibility \
        --visibility_aware_culling \
        --compare_center_patch \
        --use_similarity_grouping \
        --closest_n_views ${closest_n_views} \
        --n_clusters ${n_clusters}
    
    # Step 3: Training with SIMILARITY GROUPING (using train_similarity.py)
    echo ""
    echo "Step 2/4: Training with SIMILARITY GROUPING (dynamic similarity-based groups + DS + Lap losses)..."
    python train_similarity.py \
        -s "${augmented_path}/${scene}" \
        -m "${exp_path}/${exp_name}/${scene}" \
        -i ${images_folder} \
        --eval \
        --bundle_training \
        --camera_order covisibility \
        --enable_ds_lap \
        --lambda_ds 1.2 \
        --lambda_lap 0.4 \
        --n_clusters ${n_clusters}
    
    # Step 4: Rendering
    echo ""
    echo "Step 3/4: Rendering test views..."
    python render.py -m "${exp_path}/${exp_name}/${scene}" --skip_train
    
    # Step 5: Compute metrics
    echo ""
    echo "Step 4/4: Computing metrics..."
    python metrics.py -m "${exp_path}/${exp_name}/${scene}"
    
    # Step 5: Save individual scene results
    python utils/experiment_utils.py ${scene} ${exp_name}
    
    echo ""
    echo "Completed scene: ${scene}"
    echo "=========================================="
done

# Generate experiment summary
echo ""
echo "=========================================="
echo "All scenes completed!"
echo "Results saved to: ${exp_path}/${exp_name}"
echo "=========================================="
