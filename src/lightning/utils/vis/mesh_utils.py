import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

def get_opengl_conversion_matrix():
    """
    Returns a transformation matrix to convert from camera coordinates to OpenGL coordinates (from colmap/opencv).
    Adjust this based on your coordinate system needs.
    """
    # Example: Flip the Y and Z axes
    return np.array([
        [-1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0,  0, 1]
    ])

def apply_scene_alignment(
    scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray
) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler(
        "y", 180, degrees=True
    ).as_matrix()

    # Apply transformation
    initial_transformation = (
        np.linalg.inv(extrinsics_matrices[0])
        @ opengl_conversion_matrix
        @ align_rotation
    )
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a homogeneous transformation to a set of points.

    Args:
        transform (np.ndarray): A 4x4 transformation matrix.
        points (np.ndarray): An (N, 3) array of points.

    Returns:
        np.ndarray: Transformed points as an (N, 3) array.
    """
    homog_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_homog = (transform @ homog_points.T).T
    return transformed_homog[:, :3]

def compute_camera_faces(camera_cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh based on the cone shape.

    Args:
        camera_cone_shape (trimesh.Trimesh): The cone representing the camera.

    Returns:
        np.ndarray: Array of face indices.
    """
    # This is a placeholder. Implement face computation based on camera_cone_shape
    return camera_cone_shape.faces

def colormap(value: float):
    """
    Example colormap function that returns RGBA values.
    Modify this to use a proper colormap as needed.
    """
    return (value, 0.5, 1 - value, 1.0)

def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler(
        "z", 45, degrees=True
    ).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler(
        "z", 2, degrees=True
    ).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(
        complete_transform, vertices_combined
    )

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Adjust face indices
    mesh_faces += len(camera_cone_shape.vertices) * 0  # Modify as needed

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(
        vertices=vertices_transformed, faces=mesh_faces
    )
    camera_mesh.visual.face_colors = face_colors
    scene.add_geometry(camera_mesh)