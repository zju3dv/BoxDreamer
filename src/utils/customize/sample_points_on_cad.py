import open3d as o3d
import os.path as osp
import numpy as np
from plyfile import PlyData
import trimesh
def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def sample_points_on_cad(cad_model, n_num=1000, save_p3d_path=None):
    """
    cad_model: str(path) or open3d mesh
    """
    if isinstance(cad_model, str):
        assert osp.exists(cad_model), f"CAD model mesh: {cad_model} not exists"
        mesh = o3d.io.read_triangle_mesh(cad_model)
    else:
        mesh = cad_model

    model_corners = get_model_corners(np.asarray(mesh.vertices))
    model_center = (np.max(model_corners, 0, keepdims=True) + np.min(model_corners, 0, keepdims=True)) / 2
    model_8corners_center = np.concatenate([model_corners, model_center], axis=0) # 9*3

    # Sample uniformly
    sampled_3D_points = mesh.sample_points_uniformly(n_num)

    # Save:
    if save_p3d_path is not None:
        o3d.io.write_point_cloud(save_p3d_path, sampled_3D_points)

    sampled_3D_points = np.asarray(sampled_3D_points.points)
    return sampled_3D_points.astype(np.float32), model_8corners_center.astype(np.float32) # 9*3

def load_points_from_cad(cad_model, max_num=-1, save_p3d_path=None):
    """
    cad_model: str(path) or open3d mesh
    """
    if isinstance(cad_model, str):
        assert osp.exists(cad_model), f"CAD model mesh: {cad_model} not exists"
        mesh = o3d.io.read_triangle_mesh(cad_model)
    else:
        mesh = cad_model

    model_corners = get_model_corners(np.asarray(mesh.vertices))
    model_center = (np.max(model_corners, 0, keepdims=True) + np.min(model_corners, 0, keepdims=True)) / 2
    model_8corners_center = np.concatenate([model_corners, model_center], axis=0) # 9*3

    # Sample uniformly
    # sampled_3D_points = o3d.geometry.sample_points_uniformly(mesh, n_num)
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] > max_num and max_num != -1:
        sampled_3D_points = mesh.sample_points_uniformly(max_num)
        vertices = np.asarray(sampled_3D_points.points)

    # Save:
    if save_p3d_path is not None:
        o3d.io.write_point_cloud(save_p3d_path, vertices)

    return vertices.astype(np.float32), model_8corners_center.astype(np.float32) # 9*3

def model_diameter_from_bbox(bbox):
    """
    bbox: 8*3 or 9*3(including center at last row)
    """
    min_coord = bbox[0] # 3
    max_coord = bbox[7] # 3
    diameter = np.linalg.norm(max_coord - min_coord)
    return diameter

def fast_mesh_to_point_cloud(mesh_path, num_points=10000):
    """
    Convert a 3D mesh to a point cloud using fast uniform surface sampling.
    Handles both Trimesh and Scene objects.

    Parameters:
        mesh_path (str): Path to the input mesh file (.obj, .glb, etc.).
        num_points (int): Number of points to sample from the mesh surface.

    Returns:
        numpy.ndarray: A (num_points, 3) array of sampled point cloud coordinates.
    """
    # Load the mesh or scene
    mesh = trimesh.load(mesh_path)

    # If the object is a Scene, merge all geometries into a single Trimesh
    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries in the scene into a single Trimesh
        combined_mesh: trimesh.Trimesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )
    else:
        # If it's already a Trimesh, use it directly
        combined_mesh: trimesh.Trimesh = mesh

    # Sample points from the combined mesh
    sampled_points = combined_mesh.sample(num_points)

    return sampled_points

def normalize_point_cloud(points):
    """
    Normalize the point cloud to center it at the origin and scale it to fit within a unit bounding box.

    Parameters:
        points (numpy.ndarray): A (N, 3) array of point cloud coordinates.

    Returns:
        numpy.ndarray: The normalized point cloud.
    """
    # Compute the bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    
    diag = np.linalg.norm(bbox_max - bbox_min)

    # Compute the center (offset) to translate the point cloud to the origin
    offset = -(bbox_min + bbox_max) / 2
    points += offset

    # Recompute the bounding box after translation
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    # Compute the scale factor based on the largest dimension
    # scale = 1.0 / np.max(bbox_max - bbox_min)
    # scale = 1.0 / (diag / 2)  # Add a small margin 
    
    # use max min normalization to -1, 1
    scale = 2 / diag
    
    # Scale the point cloud
    points *= scale

    return points

def get_all_points_on_model(cad_model_path, return_color=False, max_size=-1):
    if cad_model_path.endswith('.ply'):
        ply = PlyData.read(cad_model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        if model.shape[0] > max_size and max_size != -1:
            # uniformly downsample
            idx = np.random.choice(np.arange(model.shape[0]), max_size, replace=False)
            model = model[idx]
        if return_color:
            r = data['red']
            g = data['green']
            b = data['blue']
            color = np.stack([r, g, b], axis=-1)
            if color.shape[0] > max_size and max_size != -1:
                color = color[idx]
            return model, color
    elif cad_model_path.endswith('.glb') or cad_model_path.endswith('.obj'):
        sampled_3D_points = fast_mesh_to_point_cloud(cad_model_path, num_points=10000)
        normalized_point_cloud = normalize_point_cloud(sampled_3D_points)
        model = normalized_point_cloud
    elif cad_model_path.endswith('.xyz'):
        model = np.loadtxt(cad_model_path)
    else:
        raise NotImplementedError(f"Model format {cad_model_path} not implemented")
                                  
    return model