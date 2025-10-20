import argparse
import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil

from utils.colmap_utils import *
from utils.bundle_utils import cluster_cameras
from utils.aug_utils import *
from utils.scheduler_utils import ImageClustering


def augment(colmap_path, image_path, augment_path, camera_order, visibility_aware_culling, compare_center_patch, n_clusters, use_similarity_grouping=False, closest_n_views=12):
    colmap_images, colmap_points3D, colmap_cameras = get_colmap_data(colmap_path)
    np.seterr(divide='ignore', invalid='ignore')
    sorted_keys = cluster_cameras(colmap_path, camera_order, n_clusters=n_clusters)
    
    # Setup similarity-based view matching if enabled
    if use_similarity_grouping:
        image_clustering = ImageClustering(dataset_path=colmap_path, n_clusters=n_clusters)
        # Build closest_n_views dictionary: image_id -> list of closest image_ids
        name_to_id = {colmap_images[key].name: key for key in sorted_keys}
        closest_views_dict = {}
        for view_id in sorted_keys:
            view_name = colmap_images[view_id].name
            similarities = [(name, sim) for name, sim in image_clustering.W_dict[view_name].items()]
            similarities = [s for s in similarities if s[0] != view_name]
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            closest_names = [name for name, _ in sorted_similarities[:closest_n_views]]
            closest_views_dict[view_id] = [name_to_id[name] for name in closest_names]
    else:
        closest_views_dict = None

    points3d = []
    points3d_rgb = []
    for key in sorted(colmap_points3D.keys()):
        points3d.append(colmap_points3D[key].xyz)
        points3d_rgb.append(colmap_points3D[key].rgb)
    points3d = np.array(points3d)
    points3d_rgb = np.array(points3d_rgb)

    image_sample = cv2.imread(os.path.join(image_path, colmap_images[sorted_keys[0]].name))
    intrinsics_camera = compute_intrinsics(colmap_cameras, image_sample.shape[1], image_sample.shape[0])
    rotations_image, translations_image = compute_extrinsics(colmap_images)

    count = 0
    roots = {}
    pbar = tqdm(range(len(sorted_keys)))
    for view_idx in pbar:
        view = sorted_keys[view_idx]
        view_root, augmented_count = image_quadtree_augmentation(
            view,
            image_path,
            colmap_cameras,
            colmap_images,
            colmap_points3D,
            points3d,
            points3d_rgb,
            intrinsics_camera,
            rotations_image,
            translations_image,
            visibility_aware_culling=visibility_aware_culling,
        )
        count += augmented_count
        pbar.set_description(f"{count} points augmented")
        roots[view] = view_root

    for view1_idx in tqdm(range(len(sorted_keys))):
        view1 = sorted_keys[view1_idx]
        
        # Select reference views based on similarity or sequential neighbors
        if use_similarity_grouping and closest_views_dict:
            # Use similarity-based closest views
            ref_views = closest_views_dict[view1]
        else:
            # Use sequential neighbors (original behavior)
            ref_view_offsets = [6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6]
            ref_views = []
            for offset in ref_view_offsets:
                view2_idx = view1_idx + offset
                if view2_idx > len(sorted_keys) - 1:
                    view2_idx = view2_idx - len(sorted_keys)
                elif view2_idx < 0:
                    view2_idx = len(sorted_keys) + view2_idx
                ref_views.append(sorted_keys[view2_idx])
        
        for view2 in ref_views:
            view1_root = roots[view1]
            view2_root = roots[view2]

            image_view2 = cv2.imread(os.path.join(image_path, colmap_images[view2].name))

            view1_sample_points_world, view1_sample_points_rgb = transform_sample_3d(view1,
                                                                                     view1_root,
                                                                                     colmap_images,
                                                                                     colmap_cameras,
                                                                                     intrinsics_camera,
                                                                                     rotations_image,
                                                                                     translations_image)
            view1_sample_points_view2, view1_sample_points_view2_depth = project_3d_to_2d(view1_sample_points_world,
                                                                                          intrinsics_camera[colmap_images[view2].camera_id],
                                                                                          np.concatenate((np.array(rotations_image[view2]),
                                                                                                          np.array(translations_image[view2]).reshape(3,1)),
                                                                                                          axis=1))
            points3d_view2_pixcoord, points3d_view2_depth = project_3d_to_2d(points3d,
                                                                             intrinsics_camera[colmap_images[view2].camera_id],
                                                                             np.concatenate((np.array(rotations_image[view2]),
                                                                                             np.array(translations_image[view2]).reshape(3,1)),
                                                                                             axis=1))
            
            matching_log = []
            for i in range(view1_sample_points_world.shape[0]):
                x, y = view1_sample_points_view2[i]
                corresponding_node_type = None
                error = None
                
                # Case 1: Culling
                if (view1_sample_points_view2_depth[i] < 0) | \
                   (view1_sample_points_view2[i, 0] < 0) | \
                   (view1_sample_points_view2[i, 0] >= image_view2.shape[1]) | \
                   (view1_sample_points_view2[i, 1] < 0) | \
                   (view1_sample_points_view2[i, 1] >= image_view2.shape[0]) | \
                   np.isnan(view1_sample_points_view2[i]).any(axis=0):
                    corresponding_node_type = "culled"
                    matching_log.append([view2, corresponding_node_type, error])
                    continue
                
                # Case 2: Find corresponding node
                view2_corresponding_node = find_leaf_node(view2_root, x, y)
                if view2_corresponding_node is None:
                    corresponding_node_type = "missing"
                    matching_log.append([view2, corresponding_node_type, error])
                    continue
                
                # Case 3: Process unoccupied node
                if view2_corresponding_node.unoccupied:
                    if view2_corresponding_node.depth_interpolated:
                        error = np.linalg.norm(view1_sample_points_view2_depth[i] - view2_corresponding_node.sampled_point_depth)
                        if error < 0.2 * view2_corresponding_node.sampled_point_depth:
                            if compare_center_patch:
                                try:
                                    view1_sample_point_patch = image_view2[int(view1_sample_points_view2[i, 1])-1:\
                                                                           int(view1_sample_points_view2[i,1])+2,
                                                                           int(view1_sample_points_view2[i, 0])-1:\
                                                                           int(view1_sample_points_view2[i,0])+2]
                                    view2_corresponding_node_patch = image_view2[int(view2_corresponding_node.sampled_point_uv[1])-1:\
                                                                           int(view2_corresponding_node.sampled_point_uv[1])+2,
                                                                           int(view2_corresponding_node.sampled_point_uv[0])-1:\
                                                                           int(view2_corresponding_node.sampled_point_uv[0])+2]
                                    if compare_local_texture(view1_sample_point_patch, view2_corresponding_node_patch) > 0.5:
                                        corresponding_node_type = "sampledrejected"
                                    else:
                                        corresponding_node_type = "sampled"
                                except IndexError:
                                    corresponding_node_type = "sampledrejected"
                            else:
                                corresponding_node_type = "sampled"
                        else:
                            corresponding_node_type = "sampledrejected"
                    else:
                        corresponding_node_type = "depthrejected"
                else:
                    corresponding_3d_depth = np.array(view2_corresponding_node.points3d_depths)
                    error = np.linalg.norm(view1_sample_points_view2_depth[i] - corresponding_3d_depth)

                    if np.min(error) < 0.2 * corresponding_3d_depth[np.argmin(error)]:
                        if compare_center_patch:
                            try:
                                point_3d_coord = points3d_view2_pixcoord[view2_corresponding_node.points3d_indices[np.argmin(error)]]
                                point_3d_patch = image_view2[int(point_3d_coord[1])-1:\
                                                             int(point_3d_coord[1])+2,
                                                             int(point_3d_coord[0])-1:\
                                                             int(point_3d_coord[0])+2]
                                view1_sample_point_patch = image_view2[int(view1_sample_points_view2[i, 1])-1:\
                                                                       int(view1_sample_points_view2[i,1])+2,
                                                                       int(view1_sample_points_view2[i, 0])-1:\
                                                                       int(view1_sample_points_view2[i,0])+2]
                                if compare_local_texture(view1_sample_point_patch, point_3d_patch) > 0.5:
                                    corresponding_node_type = "rejectedoccupied3d"
                                else:
                                    corresponding_node_type = "occupied3d"
                            except IndexError:
                                corresponding_node_type = "rejectedoccupied3d"
                        else:
                            corresponding_node_type = "occupied3d"
                    else:
                        corresponding_node_type = "rejectedoccupied3d"
                
                # 모든 경우에 대해 로그 추가
                matching_log.append([view2, corresponding_node_type, error])

            node_index = 0
            view1_leaf_nodes = []
            gather_leaf_nodes(view1_root, view1_leaf_nodes)
            for node in view1_leaf_nodes:
                if node.unoccupied:
                    if node.depth_interpolated:
                        node.matching_log[view2] = matching_log[node_index]
                        if matching_log[node_index][1] in ["depthrejected", "missing", "culled"]:
                            None
                        else:
                            node.inference_count += 1
                        node.rejection_count += 1 if matching_log[node_index][1] in ["rejectedoccupied3d",
                                                                                    "sampledrejected"] else 0
                        node_index += 1

    sampled_points_total = []
    sampled_points_rgb_total = []
    sampled_points_uv_total = []
    sampled_points_neighbors_uv_total = []
    for view in sorted_keys:
        view_root = roots[view]
        leaf_nodes = []
        gather_leaf_nodes(view_root, leaf_nodes)
        for node in leaf_nodes:
            if node.unoccupied:
                if node.depth_interpolated:
                    if node.inference_count > 0:
                        if node.inference_count - node.rejection_count >= 1:
                            sampled_points_total.append([node.sampled_point_world])
                            sampled_points_rgb_total.append([node.sampled_point_rgb])
                            sampled_points_uv_total.append([node.sampled_point_uv])
                            sampled_points_neighbors_uv_total.append([node.sampled_point_neighbours_uv])
    print("total_Sampled_points: ", len(sampled_points_total))
    xyz = np.concatenate(sampled_points_total, axis=0)
    rgb = np.concatenate(sampled_points_rgb_total, axis=0)

    last_index = write_points3D_colmap_binary(colmap_points3D, xyz, rgb, augment_path)
    print("last_index: ", last_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--augment_path", type=str, required=True)
    parser.add_argument("--camera_order", type=str, required=True, default="covisibility")
    parser.add_argument("--visibility_aware_culling", 
                   action="store_true",
                   default=False)
    parser.add_argument("--compare_center_patch", 
                   action="store_true",
                   default=False)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--use_similarity_grouping",
                   action="store_true",
                   default=False,
                   help="Use similarity-based view matching instead of sequential neighbors")
    parser.add_argument("--closest_n_views", type=int, default=12,
                   help="Number of closest views to use for similarity-based matching")
    args = parser.parse_args()
    print("args.colmap_path", args.colmap_path)
    print("args.image_path", args.image_path)
    print("args.augment_path", args.augment_path)
    print("args.camera_order", args.camera_order)
    print("args.visibility_aware_culling", args.visibility_aware_culling)
    print("args.compare_center_patch", args.compare_center_patch)
    print("args.n_clusters", args.n_clusters)
    print("args.use_similarity_grouping", args.use_similarity_grouping)
    print("args.closest_n_views", args.closest_n_views)
    augment(args.colmap_path, args.image_path, args.augment_path, args.camera_order, 
            args.visibility_aware_culling, args.compare_center_patch, args.n_clusters,
            args.use_similarity_grouping, args.closest_n_views)