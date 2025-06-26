# this script is used to visualize the results of the ours vs baseline

import os
import glob
import cv2
import numpy as np
from PIL import Image
from src.lightning.utils.vis.vis_utils import *
from tqdm import tqdm
import pickle
from src.utils.customize.sample_points_on_cad import get_all_points_on_model
import json

onepose_lm_mapping = {
    'ape': '0801-lm_o1-others',
    'can': '0805-lm_o5-others',
    'cat': '0806-lm_o6-others',
    'driller': '0808-lm_o8-others',
    'duck': '0809-lm_o9-others',
    'eggbox': '0810-lm_o10-others',
    'glue': '0811-lm_o11-others',
    'holepuncher': '0812-lm_o12-others',
}
onepose_ycbv_mapping = {
    '002_master_chef_can': '0802-ycbv2-others',
    '003_cracker_box': '0803-ycbv3-others',
    '004_sugar_box': '0804-ycbv4-others',
    '005_tomato_soup_can': '0805-ycbv5-others',
    '006_mustard_bottle': '0806-ycbv6-others',
    '007_tuna_fish_can': '0807-ycbv7-others',
    '008_pudding_box': '0808-ycbv8-others',
    '009_gelatin_box': '0809-ycbv9-others',
    '010_potted_meat_can': '0810-ycbv10-others',
    '011_banana': '0811-ycbv11-others',
    '019_pitcher_base': '0819-ycbv19-others',
    '021_bleach_cleanser': '0821-ycbv21-others',
    '024_bowl': '0824-ycbv24-others',
    '025_mug': '0825-ycbv25-others',
    '035_power_drill': '0835-ycbv35-others',
    '036_wood_block': '0836-ycbv36-others',
    '037_scissors': '0837-ycbv37-others',
    '040_large_marker': '0840-ycbv40-others',
    '051_large_clamp': '0851-ycbv51-others',
    '052_extra_large_clamp': '0852-ycbv52-others',
    '061_foam_brick': '0861-ycbv61-others',
}
ours_ycbv_mapping = {
    '002': '002_master_chef_can',
    '003': '003_cracker_box',
    '004': '004_sugar_box',
    '005': '005_tomato_soup_can',
    '006': '006_mustard_bottle',
    '007': '007_tuna_fish_can',
    '008': '008_pudding_box',
    '009': '009_gelatin_box',
    '010': '010_potted_meat_can',
    '011': '011_banana',
    '019': '019_pitcher_base',
    '021': '021_bleach_cleanser',
    '024': '024_bowl',
    '025': '025_mug',
    '035': '035_power_drill',
    '036': '036_wood_block',
    '037': '037_scissors',
    '040': '040_large_marker',
    '051': '051_large_clamp',
    '052': '052_extra_large_clamp',
    '061': '061_foam_brick',
}
ycbv_onepose_mapping = {v: k for k, v in onepose_ycbv_mapping.items()}
data_root_mapping ={
    'lmo': '/home/yyh/lab/OnePoseV3/data/lmo',
    'ycbv': '/home/yyh/lab/OnePoseV3/data/ycbv'
}
lm_onepose_mapping = {v: k for k, v in onepose_lm_mapping.items()}

asset_paths = {
    'lmo': {
        'gen6d': '/home/yyh/lab/OnePoseV3/cache/baseline_cache/lmo-full/agg_save_pose_gen6d/occluded_linemod/gen6d_pretrain_all_uniform_refine3_gt0',
        'onepose': '/home/yyh/lab/OnePoseV3/cache/baseline_cache/lmo-full/agg_save_pose_oneposeplus/OCCLUDED_LINEMOD_dataset/inference_OCCLUDED_LINEMOD_from_lm_-1_uniform',
        'ours': '/home/yyh/lab/OnePoseV3/logs/runs/2025-02-28/20-56-41-heatmap_v2_wreal_hard_72ep-lmo-25frame'
    },
    'ycbv': {
        'gen6d': '/home/yyh/lab/OnePoseV3/cache/baseline_cache/ycbv/mo/gen6d_pretrain_25_fps_refine3_gt1',
        'onepose': '/home/yyh/lab/OnePoseV3/cache/baseline_cache/ycbv/mo/inference_ycbv_most_overlapping_25_fps',
        'ours': '/home/yyh/lab/OnePoseV3/logs/runs/2025-02-28/11-15-35-heatmap_v2_wreal_88ep-ycbv-mo-5frame'
    }
}


def gen6d_parser(path: str, dataset: str):
    # from pre dumped file load predictions
    objs = os.listdir(path)
    obj_pose = {}
    for obj in tqdm(objs, desc="reading gen6d results", total=len(objs)):
        if not os.path.isdir(os.path.join(path, obj)):
            continue
        tgt_file = os.path.join(path, obj, f"{obj}.pkl")
        # load the file
        with open(tgt_file, 'rb') as f:
            data = pickle.load(f)
        
        # data is two dim list [b, idx]
        poses = data
        ## flatten the poses
        poses = np.concatenate(poses, axis=0)
        print(obj, ": ", poses.shape)
        # log the shape
        if dataset == 'lmo':
            obj_pose[obj] = poses
        elif dataset == 'ycbv':
            obj_pose[ycbv_onepose_mapping[obj]] = poses
        else:
            raise ValueError("dataset not supported")
        
    # load json file from path
    # find the json file from the path
    json_file = glob.glob(os.path.join(path, "*.json"))
    assert len(json_file) == 1, "json file not found"
    # load the json file
    with open(json_file[0], 'r') as f:
        data = json.load(f)
    
    # the dict format is like:
    # category: {idx: path}
    # convert to category: {path: idx}, and only keep path's basename without extension
    path_dict = {}
    for cat, idx_path in data.items():
        if dataset == 'lmo':
            path_dict[cat] = {os.path.basename(path).split('.')[0]: idx for idx, path in idx_path.items()}
        else:
            # another format: "origin_data/ycbv/test/002_master_chef_can/0048/000000-color.png"
            # concat seq and frame: 0048-000000
            path_dict[cat] = {f'{path.split("/")[-2]}-{path.split("/")[-1].split("-")[0]}': idx for idx, path in idx_path.items()}
        
        
    return obj_pose, path_dict

def onepose_parser(path: str, dataset: str):
    # from pre dumped file load predictions
    objs = os.listdir(path)
    obj_pose = {}
    for obj in tqdm(objs, desc="reading onepose results", total=len(objs)):
        if not os.path.isdir(os.path.join(path, obj)):
            continue
        tgt_file = os.path.join(path, obj, f"{obj}.pkl")
        # load the file
        # if file not found, skip
        if not os.path.exists(tgt_file):
            continue
        
        with open(tgt_file, 'rb') as f:
            data = pickle.load(f)
        
        # data is two dim list [b, idx]
        poses = data
        ## flatten the poses
        poses = np.array(poses)
        # print(lm_onepose_mapping[obj], ": ", poses.shape)
        
        # log the shape
        if dataset == 'lmo':
            obj_pose[lm_onepose_mapping[obj]] = poses
        elif dataset == 'ycbv':
            obj_pose[ycbv_onepose_mapping[obj]] = poses
        else:
            raise ValueError("dataset not supported")
        
    # load json file from path
    # find the json file from the path
    json_file = glob.glob(os.path.join(path, "*.json"))
    assert len(json_file) == 1, "json file not found"
    # load the json file
    with open(json_file[0], 'r') as f:
        data = json.load(f)
    
    # the dict format is like:
    # category: {idx: path}
    # convert to category: {path: idx}, and only keep path's basename without extension
    path_dict = {}
    for cat, idx_path in data.items():
        if dataset == 'lmo':
            path_dict[cat] = {os.path.basename(path).split('.')[0]: idx for idx, path in idx_path.items()}
        else:
            # another format: "origin_data/ycbv/test/002_master_chef_can/0048/000000-color.png"
            # concat seq and frame: 0048-000000
            path_dict[cat] = {f'{path.split("/")[-2]}-{path.split("/")[-1].split("-")[0]}': idx for idx, path in idx_path.items()}
            
        
        
    return obj_pose, path_dict

def ours_parser(path: str, dataset: str):
    # load path_error_dict_0.npy and path_pose_dict_0.npy
    path_error = np.load(path + f"/path_error_dict_0.npy", allow_pickle=True)
    path_pose = np.load(path + f"/path_pose_dict_0.npy", allow_pickle=True)

    if isinstance(path_pose, np.ndarray):
        path_pose = path_pose.item()
    if isinstance(path_error, np.ndarray):
        path_error = path_error.item()
    
    # cat: path: pose or cat: path: pose rotation error
    
    if dataset == 'ycbv':
        # replace the keys
        new_path_error = {}
        new_path_pose = {}
        
        for cat, path_dict in path_error.items():
            if cat == 'all':
                continue
            new_path_error[ours_ycbv_mapping[cat]] = path_dict
        
        for cat, path_dict in path_pose.items():
            if cat == 'all':
                continue
            new_path_pose[ours_ycbv_mapping[cat]] = path_dict
            
        path_error = new_path_error
        path_pose = new_path_pose
            
    return path_error, path_pose


def handler(gen6d, gen6d_mapping, onepose, onepose_mapping, ours_pp, ours_pe, obj=None, topk=10, dataset='lmo'):
    """
    Compare our method's top K results with baseline methods.
    
    Args:
        gen6d: Dict mapping object names to pose arrays from Gen6D
        onepose: Dict mapping object names to pose arrays from OnePose
        ours_pp: Nested dict {obj_name: {path: pose}} from our method
        ours_pe: Nested dict {obj_name: {path: rot_error}} from our method
        obj: Specific object to analyze (if None, analyze all objects)
        topk: Number of top results to compare
    
    Returns:
        Dict with comparison results
    """
    # Convert numpy arrays to dictionaries if they're not already
    if isinstance(ours_pp, np.ndarray):
        ours_pp = ours_pp.item()
    if isinstance(ours_pe, np.ndarray):
        ours_pe = ours_pe.item()
        
    
    # Determine objects to process
    objects_to_process = [obj] if obj is not None else list(ours_pe.keys())
    
    results = {}
    
    for current_obj in objects_to_process:
        if current_obj not in gen6d or current_obj not in onepose:
            print(f"Object {current_obj} not found in baseline results. Skipping.")
            continue
        
        if current_obj not in ours_pe or current_obj not in ours_pp:
            print(f"Object {current_obj} not found in our results. Skipping.")
            continue
        
        print(f"\nProcessing object: {current_obj}")
        
        # Get all paths and their errors for this object
        obj_errors = ours_pe[current_obj]
        
        # Sort paths by rotation error (ascending)
        sorted_paths = sorted(obj_errors.keys(), key=lambda p: obj_errors[p], reverse=False)
        # sorted_paths = list(obj_errors.keys())
        
        # Take top K paths with lowest errors
        top_paths = sorted_paths[:min(topk, len(sorted_paths))]
        
        # Create mapping from path to index in baseline arrays
        # We need to determine which index in baseline arrays corresponds to each path
        all_paths = sorted(obj_errors.keys())
        path_to_index = {path: i for i, path in enumerate(all_paths)}
        
        # Store results for this object
        obj_results = []
        
        print(f"Top {len(top_paths)} paths with lowest rotation errors for {current_obj}:")
        
        for i, path in enumerate(top_paths):
            # Get our method's results
            rotation_error = obj_errors[path]
            our_pose = ours_pp[current_obj][path]
            
            # Get index in baseline arrays
            # baseline_idx = path_to_index[path]
            try:
                if dataset == 'lmo':
                    gen6d_index = gen6d_mapping[current_obj][os.path.basename(path).split('.')[0].split('-')[0]]
                    onepose_index = onepose_mapping[current_obj][os.path.basename(path).split('.')[0].split('-')[0]]
                elif dataset == 'ycbv':
                    # if path.split("/")[-2] != '0054':
                    #     raise KeyError
                    gen6d_index = gen6d_mapping[current_obj][f'{path.split("/")[-2]}-{path.split("/")[-1].split("-")[0]}']
                    onepose_index = onepose_mapping[current_obj][f'{path.split("/")[-2]}-{path.split("/")[-1].split("-")[0]}']
            except KeyError:
                print(f"Path {path} not found in baseline mapping. Skipping.")
                continue
            
            
            # Get baseline results
            gen6d_pose = gen6d[current_obj][int(gen6d_index)]
            onepose_pose = onepose[current_obj][int(onepose_index)]
            
            # print(f"  Result {i+1}:")
            # print(f"    Path: {path}")
            # print(f"    Baseline Index: {baseline_idx}")
            # print(f"    Rotation Error: {rotation_error:.4f}")
            
            # Store results for this path
            obj_results.append({
                'path': path,
                'rotation_error': rotation_error,
                'our_pose': our_pose,
                'gen6d_pose': gen6d_pose,
                'onepose_pose': onepose_pose
            })
            
        
        results[current_obj] = obj_results
    
    ret = vis_and_dump_results(results, dataset=dataset)
    print(ret)
    exit(0)


def vis_and_dump_results(results, crop_size=256, dataset='lmo'):
    """
    Visualize and save comparison results of different pose estimation methods.
    
    Args:
        results: Dictionary with results from handler function
        crop_size: Size of square crops (default: 256)
    
    Returns:
        Dictionary with paths to saved visualizations
    """
    # Data root path
    if dataset == 'lmo':
        data_root = data_root_mapping['lmo']
    elif dataset == 'ycbv':
        data_root = data_root_mapping['ycbv']
    model_root = os.path.join(data_root, "models")
    
    # Create output directory
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    all_vis_paths = {}
    
    # Process each object's results
    for obj, obj_results in results.items():
        obj_output_dir = os.path.join(output_dir, obj)
        obj_croped_output_dir = os.path.join(output_dir, obj, "croped")
        os.makedirs(obj_output_dir, exist_ok=True)
        os.makedirs(obj_croped_output_dir, exist_ok=True)
        
        print(f"\nVisualizing results for object: {obj}")
        
        # Load 3D model
        
        if dataset == 'lmo':
            model_path = os.path.join(model_root, f"{obj}/{obj}.ply")
        elif dataset == 'ycbv':
            model_path = os.path.join(model_root, f"{obj}/points.xyz")
        try:
            model_points = get_all_points_on_model(model_path)
            bbox_3d = get_3d_bbox_from_pts(model_points)  # 8x3 array of corner points
        except Exception as e:
            print(f"Error loading model for {obj}: {e}")
            continue
        
        # Prepare to store visualization paths
        vis_paths = []
        
        # Process each result for this object
        for i, result in enumerate(obj_results):
            # Get image path and related paths
            img_path = result['path']
            pose_path = img_path.replace("-color.png", "-pose.txt")
            intrinsic_path = img_path.replace("-color.png", "-intrinsics.txt")
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image from {img_path}")
                    continue
                
                # Load ground truth pose
                gt_pose = np.loadtxt(pose_path) if os.path.exists(pose_path) else None
                
                # Load camera intrinsics
                K = np.loadtxt(intrinsic_path) if os.path.exists(intrinsic_path) else np.array([
                    [572.4114, 0, 325.2611],
                    [0, 573.57043, 242.04899],
                    [0, 0, 1]
                ])
                
                # Create visualization comparing all methods
                comparison_vis = create_comparison_grid(
                    image.copy(), 
                    K, 
                    bbox_3d,
                    gt_pose=gt_pose,
                    our_pose=result['our_pose'],
                    onepose_pose=result['onepose_pose'],
                    gen6d_pose=result['gen6d_pose'],
                    rotation_error=result['rotation_error'],
                    img_path=img_path
                )
                
                # Save comparison visualization
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                vis_path = os.path.join(obj_output_dir, f"{base_filename}_comparison.jpg")
                
                cv2.imwrite(vis_path, comparison_vis)
                
                # Create and save square crops for each method
                crop_paths = create_square_crops(
                    image.copy(),
                    K,
                    bbox_3d,
                    gt_pose,
                    result['our_pose'],
                    result['onepose_pose'],
                    result['gen6d_pose'],
                    obj_croped_output_dir,
                    base_filename,
                    crop_size
                )
                
                # Store visualization path
                vis_paths.append({
                    'path': img_path,
                    'vis_path': vis_path,
                    'crop_paths': crop_paths,
                    'rotation_error': result['rotation_error']
                })
                
                print(f"  Processed {i+1}/{len(obj_results)}: {os.path.basename(img_path)}, rot_err: {result['rotation_error']:.4f}")
                
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                import traceback
                print(traceback.format_exc())
                exit()
        
        # Create a montage of all visualizations for this object
        if vis_paths:
            # montage_path = create_montage(obj_output_dir, obj, vis_paths)
            all_vis_paths[obj] = {
                'individual_vis': vis_paths,
                # 'montage': montage_path
            }
    
    return all_vis_paths

def create_square_crops(image, K, bbox_3d, gt_pose, our_pose, onepose_pose, gen6d_pose, 
                        output_dir, base_filename, crop_size=256):
    """
    Create square crops of the object for all three methods and save them.
    
    Args:
        image: Original image
        K: Camera intrinsic matrix
        bbox_3d: 3D bounding box points
        gt_pose: Ground truth pose
        our_pose: Our method's predicted pose
        onepose_pose: OnePose+ predicted pose
        gen6d_pose: Gen6D predicted pose
        output_dir: Directory to save crops
        base_filename: Base filename (usually frame ID)
        crop_size: Size of the square crop (default: 256x256)
        
    Returns:
        Dictionary with paths to saved crop files
    """
    h, w = image.shape[:2]
    crop_paths = {}
    
    # Project GT box for crop calculation
    if gt_pose is None:
        return crop_paths  # Can't crop without GT
    
    bbox_2d_gt = reproj(K, gt_pose, bbox_3d)
    
    # Calculate crop center based on GT bbox
    center_x = int(np.mean(bbox_2d_gt[:, 0]))
    center_y = int(np.mean(bbox_2d_gt[:, 1]))
    
    # Calculate crop region (square)
    # First determine bbox dimensions
    bbox_width = int(max(bbox_2d_gt[:, 0]) - min(bbox_2d_gt[:, 0]))
    bbox_height = int(max(bbox_2d_gt[:, 1]) - min(bbox_2d_gt[:, 1]))
    
    # Use max dimension plus padding for square crop
    crop_radius = max(bbox_width, bbox_height) // 2 + 50  # 50 pixel padding
    
    # Calculate crop boundaries
    min_x = max(0, center_x - crop_radius)
    min_y = max(0, center_y - crop_radius)
    
    # Ensure square crop fits within image
    crop_side = min(crop_radius * 2, w - min_x, h - min_y)
    
    # Adjust min_x and min_y if needed to maintain square
    if min_x + crop_side > w:
        min_x = w - crop_side
    if min_y + crop_side > h:
        min_y = h - crop_side
    
    # Ensure min_x and min_y are non-negative
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    
    # Extract square crop
    crop_img = image[min_y:min_y+crop_side, min_x:min_x+crop_side].copy()
    
    # Adjust camera matrix for the crop
    K_crop = K.copy()
    K_crop[0, 2] -= min_x  # Adjust principal point x
    K_crop[1, 2] -= min_y  # Adjust principal point y
    
    # Process each method
    methods = [
        ('ours', our_pose),
        ('onepose', onepose_pose),
        ('gen6d', gen6d_pose)
    ]
    
    for method_name, method_pose in methods:
        if method_pose is None:
            continue
            
        # Create a copy of the crop for this method
        method_crop = crop_img.copy()
        
        # Draw GT box
        bbox_2d_gt_crop = reproj(K_crop, gt_pose, bbox_3d)
        method_crop = draw_3d_box(method_crop, bbox_2d_gt_crop, color='g')
        
        # Draw method's prediction
        bbox_2d_pred = reproj(K_crop, method_pose, bbox_3d)
        method_crop = draw_3d_box(method_crop, bbox_2d_pred, color='r')
        
        # Add method name to the crop
        # cv2.putText(method_crop, method_name, (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Resize to desired crop size
        if method_crop.shape[0] != crop_size:
            method_crop = cv2.resize(method_crop, (crop_size, crop_size), 
                                     interpolation=cv2.INTER_AREA if method_crop.shape[0] > crop_size else cv2.INTER_LINEAR)
        
        # Save crop
        crop_path = os.path.join(output_dir, f"{base_filename}-{method_name}-cropresults.png")
        cv2.imwrite(crop_path, method_crop)
        crop_paths[method_name] = crop_path
    
    return crop_paths

def create_comparison_grid(image, K, bbox_3d, gt_pose=None, our_pose=None, 
                          onepose_pose=None, gen6d_pose=None, rotation_error=None,
                          img_path=None):
    """
    Create a 2x2 grid visualization showing each method with ground truth.
    
    Args:
        image: Original query image
        K: Camera intrinsic matrix
        bbox_3d: 3D bounding box points (8x3)
        gt_pose: Ground truth pose matrix
        our_pose: Our method's predicted pose
        onepose_pose: OnePose+ predicted pose
        gen6d_pose: Gen6D predicted pose
        rotation_error: Rotation error of our method
        img_path: Path to the image file for displaying filename
        
    Returns:
        2x2 grid image comparing all methods with ground truth
    """
    h, w = image.shape[:2]
    grid_img = np.ones((h*2, w*2, 3), dtype=np.uint8) * 255
    
    # Project GT box for later use
    bbox_2d_gt = None
    if gt_pose is not None:
        bbox_2d_gt = reproj(K, gt_pose, bbox_3d)
    
    # Cropped Ours + GT (top-left)
    if bbox_2d_gt is not None and our_pose is not None:
        # Create cropped view of our method
        cropped_img = create_cropped_result(image.copy(), bbox_2d_gt, K, bbox_3d, gt_pose, our_pose)
        grid_img[:h, :w] = cropped_img
        cv2.putText(grid_img[:h, :w], "Cropped View (Ours)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        # If we can't create a cropped view, use original image with a message
        grid_img[:h, :w] = image.copy()
        cv2.putText(grid_img[:h, :w], "Cropped view unavailable", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Our method + GT (top-right)
    our_img = image.copy()
    if gt_pose is not None:
        our_img = draw_3d_box(our_img, bbox_2d_gt, color='g')  # GT in green
    if our_pose is not None:
        bbox_2d_ours = reproj(K, our_pose, bbox_3d)
        our_img = draw_3d_box(our_img, bbox_2d_ours, color='r')  # Our prediction in red
    grid_img[:h, w:] = our_img
    cv2.putText(grid_img[:h, w:], "Ours", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if rotation_error is not None:
        rot_err_text = f"Err: {rotation_error:.2f}"
        cv2.putText(grid_img[:h, w:], rot_err_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # OnePose+ + GT (bottom-left)
    onepose_img = image.copy()
    if gt_pose is not None:
        onepose_img = draw_3d_box(onepose_img, bbox_2d_gt, color='g')  # GT in green
    if onepose_pose is not None:
        bbox_2d_onepose = reproj(K, onepose_pose, bbox_3d)
        onepose_img = draw_3d_box(onepose_img, bbox_2d_onepose, color='r')  # OnePose+ in red
    grid_img[h:, :w] = onepose_img
    cv2.putText(grid_img[h:, :w], "OnePose++", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Gen6D + GT (bottom-right)
    gen6d_img = image.copy()
    if gt_pose is not None:
        gen6d_img = draw_3d_box(gen6d_img, bbox_2d_gt, color='g')  # GT in green
    if gen6d_pose is not None:
        bbox_2d_gen6d = reproj(K, gen6d_pose, bbox_3d)
        gen6d_img = draw_3d_box(gen6d_img, bbox_2d_gen6d, color='r')  # Gen6D in red
    grid_img[h:, w:] = gen6d_img
    cv2.putText(grid_img[h:, w:], "Gen6D(w gt detection)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add filename to bottom-left corner of the entire grid
    if img_path is not None:
        filename = os.path.basename(img_path)
        text_size = cv2.getTextSize(filename, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        # Position the text in the bottom-left with some padding
        text_x = 20
        text_y = h*2 - 20  # 20 pixels from bottom
        cv2.putText(grid_img, filename, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
    return grid_img

def create_cropped_result(image, bbox_2d_gt, K, bbox_3d, gt_pose, our_pose):
    """
    Create a cropped view focused on the object for better visualization,
    maintaining the original aspect ratio.
    
    Args:
        image: Original image
        bbox_2d_gt: Projected 2D bounding box corners from ground truth
        K: Camera intrinsic matrix
        bbox_3d: 3D bounding box points
        gt_pose: Ground truth pose
        our_pose: Our method's pose
        
    Returns:
        Cropped image with GT and our prediction boxes
    """
    h, w = image.shape[:2]
    original_aspect_ratio = w / h
    padding = 50  # Larger padding to show more context
    
    # Calculate crop region from GT bbox with padding
    min_x = max(0, int(np.min(bbox_2d_gt[:, 0])) - padding)
    min_y = max(0, int(np.min(bbox_2d_gt[:, 1])) - padding)
    max_x = min(w, int(np.max(bbox_2d_gt[:, 0])) + padding)
    max_y = min(h, int(np.max(bbox_2d_gt[:, 1])) + padding)
    
    # Calculate initial crop dimensions
    crop_w = max_x - min_x
    crop_h = max_y - min_y
    
    # Ensure minimum crop size
    crop_w = max(150, crop_w)
    crop_h = max(150, crop_h)
    
    # Adjust dimensions to maintain aspect ratio of the original image
    crop_aspect_ratio = crop_w / crop_h
    
    if crop_aspect_ratio > original_aspect_ratio:
        # Crop is too wide, increase height
        new_crop_h = int(crop_w / original_aspect_ratio)
        # Center the height adjustment
        extra_h = new_crop_h - crop_h
        min_y = max(0, min_y - extra_h // 2)
        max_y = min(h, max_y + extra_h // 2)
        # Recalculate in case we hit image boundaries
        crop_h = max_y - min_y
        crop_w = int(crop_h * original_aspect_ratio)
    else:
        # Crop is too tall, increase width
        new_crop_w = int(crop_h * original_aspect_ratio)
        # Center the width adjustment
        extra_w = new_crop_w - crop_w
        min_x = max(0, min_x - extra_w // 2)
        max_x = min(w, max_x + extra_w // 2)
        # Recalculate in case we hit image boundaries
        crop_w = max_x - min_x
        crop_h = int(crop_w / original_aspect_ratio)
    
    # Final adjustment to ensure we're within image boundaries
    if min_x + crop_w > w:
        crop_w = w - min_x
        crop_h = int(crop_w / original_aspect_ratio)
    
    if min_y + crop_h > h:
        crop_h = h - min_y
        crop_w = int(crop_h * original_aspect_ratio)
    
    # Extract the cropped region
    crop_img = image[min_y:min_y+crop_h, min_x:min_x+crop_w].copy()
    
    # Adjust camera matrix for the crop
    K_crop = K.copy()
    K_crop[0, 2] -= min_x  # Adjust principal point x
    K_crop[1, 2] -= min_y  # Adjust principal point y
    
    # Draw GT and our boxes on the cropped image
    if gt_pose is not None:
        bbox_2d_gt_crop = reproj(K_crop, gt_pose, bbox_3d)
        crop_img = draw_3d_box(crop_img, bbox_2d_gt_crop, color='g')  # GT in green
    
    if our_pose is not None:
        bbox_2d_ours_crop = reproj(K_crop, our_pose, bbox_3d)
        crop_img = draw_3d_box(crop_img, bbox_2d_ours_crop, color='r')  # Our prediction in red
    
    # Resize crop to match the grid cell size if needed, maintaining aspect ratio
    if crop_img.shape[0] != h or crop_img.shape[1] != w:
        # Use INTER_AREA for downsampling and INTER_LINEAR for upsampling
        interpolation = cv2.INTER_AREA if crop_img.shape[0] > h else cv2.INTER_LINEAR
        crop_img = cv2.resize(crop_img, (w, h), interpolation=interpolation)
    
    # Add a border to make it clear this is a cropped view
    cv2.rectangle(crop_img, (0, 0), (w-1, h-1), (255, 255, 255), 2)
    
    return crop_img

def create_montage(output_dir, obj_name, vis_paths, max_per_row=3):
    """Create a grid of visualization images"""
    # Get all comparison images
    comparison_images = []
    for vis_info in vis_paths:
        img = cv2.imread(vis_info['vis_path'])
        if img is not None:
            # Add text label with filename and rotation error
            filename = os.path.basename(vis_info['path'])
            error_text = f"{filename} (Err: {vis_info['rotation_error']:.2f})"
            cv2.putText(img, error_text, (10, img.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            comparison_images.append(img)
    
    if not comparison_images:
        print(f"No comparison images found for {obj_name}")
        return None
    
    # Get image dimensions
    h, w, c = comparison_images[0].shape
    
    # Determine grid size
    n_images = len(comparison_images)
    n_cols = min(max_per_row, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create montage image
    montage = np.ones((n_rows * h, n_cols * w, c), dtype=np.uint8) * 255
    
    # Place images in grid
    for i, img in enumerate(comparison_images):
        row = i // n_cols
        col = i % n_cols
        montage[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # Save montage
    montage_path = os.path.join(output_dir, f"{obj_name}_montage.jpg")
    cv2.imwrite(montage_path, montage)
    print(f"  Created montage for {obj_name} with {n_images} images")
    
    return montage_path
    

def main(dataset='lmo', category='cat', topk=10):
    # gen6d_ret, gen6d_mapping = gen6d_parser("/home/yyh/lab/OnePoseV3/cache/baseline_cache/lmo-full/agg_save_pose_gen6d/occluded_linemod/gen6d_pretrain_all_uniform_refine3_gt0")
    # onepose_ret, onepose_mapping = onepose_parser("/home/yyh/lab/OnePoseV3/cache/baseline_cache/lmo-full/agg_save_pose_oneposeplus/OCCLUDED_LINEMOD_dataset/inference_OCCLUDED_LINEMOD_from_lm_-1_uniform")
    # ours_pe, ours_pp = ours_parser("/home/yyh/lab/OnePoseV3/logs/runs/2025-02-27/14-34-03-heatmap_v2_wreal_hardrefine_54ep_lmo_25frame")
    
    gen6d_ret, gen6d_mapping = gen6d_parser(asset_paths[dataset]['gen6d'] , dataset)
    onepose_ret, onepose_mapping = onepose_parser(asset_paths[dataset]['onepose'], dataset)
    ours_pe, ours_pp = ours_parser(asset_paths[dataset]['ours'], dataset)
    

    handler(gen6d_ret, gen6d_mapping, onepose_ret, onepose_mapping, ours_pp, ours_pe, obj=category, topk=topk, dataset=dataset)
    

if __name__ == "__main__":
    import argparse
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='dataset name')
    parser.add_argument('--category', type=str, default='cat', help='category name')
    parser.add_argument('--topk', type=int, default=10, help='top k results to compare')
    args = parser.parse_args()
    
    main(args.dataset, args.category, args.topk)
    