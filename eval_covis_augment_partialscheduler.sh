#!/bin/bash
# Evaluation script for co-visibility-based grouping with augmentation and training
# Using partialscheduler with all extra loss terms (depth smoothness + laplacian pyramid)

# Configuration - UPDATE THESE PATHS FOR YOUR SETUP
n_clusters=4
# Set these environment variables or edit directly:
# export DATA_ROOT="/path/to/your/data"
# export OUTPUT_ROOT="/path/to/your/output"
colmap_path="${DATA_ROOT:-./data}/mipnerf360"
exp_path="${OUTPUT_ROOT:-./experiments}"
exp_name="360_covis_aug_partialscheduler_full"
augmented_path="${OUTPUT_ROOT:-./output}/augmented"

# Create augmented path if it doesn't exist
mkdir -p "${augmented_path}"

echo "=========================================="
echo "Experiment: ${exp_name}"
echo "Clusters: ${n_clusters}"
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
    
    # Step 2: Augmentation with co-visibility grouping
    echo ""
    echo "Step 1/3: Running augmentation with co-visibility grouping..."
    python augment.py \
        --colmap_path "${colmap_path}/${scene}/sparse/0" \
        --image_path "${colmap_path}/${scene}/${images_folder}" \
        --augment_path "${augmented_path}/${scene}/sparse/0/points3D.bin" \
        --camera_order covisibility \
        --visibility_aware_culling \
        --compare_center_patch \
        --n_clusters ${n_clusters}
    
    # Step 2: Training with partialscheduler and all loss terms
    echo ""
    echo "Step 2/3: Training with partialscheduler (co-visibility grouping + DS + Lap losses)..."
    python train_partialscheduler.py \
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
    
    # Step 3: Rendering
    echo ""
    echo "Step 3/3: Rendering test views..."
    python render.py -m "${exp_path}/${exp_name}/${scene}" --skip_train
    
    # Step 4: Compute metrics
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
