import os
import subprocess
import shutil
import numpy as np
from PIL import Image
import trimesh
from src.reconstruction.base import BaseReconstructor
import sqlite3
import pycolmap
import torch

class COLMAPReconstructor(BaseReconstructor):
    def __init__(self, methods='COLMAP', weights=None, config=None):
        """
        Initialize COLMAPReconstructor.

        Args:
            methods (str): Reconstruction method name, default is 'COLMAP'.
            weights (dict or None): Pretrained weights or configuration (unused, kept for compatibility).
            config (dict): Configuration dictionary, should include the following keys:
                - cache_path (str): Cache path for storing intermediate results and outputs.
                - colmap_executable (str): Path to the COLMAP executable. If it's in the system PATH, it can be omitted.
        """
        super().__init__(methods)
        self.weights = weights  # Not used in COLMAP, kept for compatibility
        self.cache_path = config.get('cache_path', './cache/colmap_cache')
        self.colmap_executable = config.get('colmap_executable', 'colmap')  # Defaults to assuming COLMAP is in PATH

        # Define necessary paths
        self.project_path = os.path.join(self.cache_path, "project")
        self.image_dir = os.path.join(self.project_path, "images")
        self.sparse_path = os.path.join(self.project_path, "sparse")
        self.database_path = os.path.join(self.project_path, "database.db")


        # Clean up previous database and project files (if they exist)
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
        if os.path.exists(self.sparse_path):
            shutil.rmtree(self.sparse_path)
            os.makedirs(self.sparse_path, exist_ok=True)
            
        # Remove the full cache path
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)
            
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(self.sparse_path, exist_ok=True)
        
    def _reinit(self):
        # Clean up previous database and project files (if they exist)
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
        if os.path.exists(self.sparse_path):
            shutil.rmtree(self.sparse_path)
            os.makedirs(self.sparse_path, exist_ok=True)
            
        # Remove the full cache path
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)
            
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(self.sparse_path, exist_ok=True)

    def _square_bbox(self, bbox: np.ndarray, padding: float = 0.1, astype=None) -> np.ndarray:
        """
        Calculate a square bounding box with optional padding.

        Args:
            bbox (np.ndarray): Bounding box in the format [x_min, y_min, x_max, y_max].
            padding (float, optional): Padding factor. Defaults to 0.1.
            astype (type, optional): Data type of the output array. Defaults to None.

        Returns:
            np.ndarray: Square bounding box in the format [x_min, y_min, x_max, y_max].
        """
        if bbox is None:
            return None
        if astype is None:
            astype = type(bbox[0])
        bbox = np.array(bbox)
        center = (bbox[:2] + bbox[2:]) / 2
        extents = (bbox[2:] - bbox[:2]) / 2
        size = max(extents) * (1 + padding)
        square_bbox = np.array(
            [center[0] - size, center[1] - size, center[0] + size, center[1] + size],
            dtype=astype,
        )
        return square_bbox            

    def _prepare_before_run(self):
        """
        Process images by applying masks or bounding boxes and save them to the cache directory.

        Returns:
            list: List of processed image paths.
        """
        assert self.images is not None, "Please set the data first"

        processed_image_paths = []

        if isinstance(self.images, (torch.Tensor, np.ndarray)):
            # Process image tensors from a data loader
            bs = self.images.shape[0]
            for i in range(bs):
                img = self.images[i]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # Convert from C, H, W to H, W, C
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)

                # Save to cache
                img_path = os.path.join(self.image_dir, f"image_{i}.jpg")
                img_pil.save(img_path)
                processed_image_paths.append(img_path)

        else:
            # Process a list of image file paths
            for img_path, mask_path in zip(self.images, self.masks):
                img = Image.open(img_path).convert('RGB')

                if mask_path and os.path.exists(mask_path):
                    if mask_path.endswith(('.png', '.jpg', '.jpeg')):
                        # If it's an image mask
                        mask = Image.open(mask_path).convert('L')
                        img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), mask)
                        bbox = np.array(mask.getbbox())
                        bbox = self._square_bbox(bbox, padding=0.1)
                        img = img.crop(bbox)

                    elif mask_path.endswith('.txt'):
                        # If it's a text-format bounding box
                        bbox = np.loadtxt(mask_path)
                        try:
                            bbox = self._square_bbox(bbox, padding=0.1, astype=int)
                        except:
                            # If bounding box format is [x0, y0, w, h]
                            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                            bbox = self._square_bbox(np.array(bbox), padding=0.1, astype=int)
                        img = img.crop(bbox)
                    else:
                        raise ValueError("Invalid mask or bounding box file")

                # Save to cache
                processed_img_path = os.path.join(self.image_dir, os.path.basename(img_path))
                img.save(processed_img_path)
                processed_image_paths.append(processed_img_path)

        return processed_image_paths

    def run(self):
        """
        Execute the COLMAP reconstruction pipeline.
        """
        self._reinit()
        assert self.images is not None, "Please set the data first"
        assert hasattr(self, 'gt_poses'), "Please set ground truth poses (self.gt_poses) first"
        assert hasattr(self, 'intinsics'), "Please set camera intrinsics (self.intinsics) first"
        assert len(self.gt_poses) == len(self.intinsics) == len(self.images), "Mismatch in the number of poses, intrinsics, and images."
        try:
            # Step 1: Process images
            print("Processing images...")
            image_paths = self._prepare_before_run()
            print(f"Processed {len(image_paths)} images.")

            # Step 2: Initialize the database, insert camera and image information
            print("Initializing COLMAP database with known poses and intrinsics...")
            self._initialize_database(image_paths, self.gt_poses, self.intinsics)
            print("Database initialization completed.")

            # Step 3: Feature extraction
            print("Extracting features with COLMAP...")
            self._run_feature_extractor()
            self._check_database('keypoints')
            self._check_database('descriptors')
            print("Feature extraction completed.")

            # Step 4: Feature matching
            print("Matching features with COLMAP...")
            self._run_exhaustive_matcher()
            self._check_database('matches')
            print("Feature matching completed.")

            # Step 5: Sparse reconstruction
            print("Performing sparse reconstruction with COLMAP...")
            self._run_colmap_reconstruction()
            print("Sparse reconstruction completed.")


            # Step 6: Export point cloud
            print("Exporting point cloud to PLY...")
            path = self._export_point_cloud()
            print("Point cloud exported.")
            
            
            # apply pruning
            path = self._self_pruning(path)
            return path
            
        except Exception as e:
            print(f"COLMAP reconstruction failed: {e}")
            return 'none'

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion (qw, qx, qy, qz).

        Args:
            R (np.ndarray): 3x3 rotation matrix.

        Returns:
            list: Quaternion [qw, qx, qy, qz].
        """
        # Use numpy to compute the quaternion
        Q = np.empty((4, ))
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            Q[0] = 0.25 / s
            Q[1] = (R[2, 1] - R[1, 2]) * s
            Q[2] = (R[0, 2] - R[2, 0]) * s
            Q[3] = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            Q[0] = (R[2, 1] - R[1, 2]) / s
            Q[1] = 0.25 * s
            Q[2] = (R[0, 1] + R[1, 0]) / s
            Q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            Q[0] = (R[0, 2] - R[2, 0]) / s
            Q[1] = (R[0, 1] + R[1, 0]) / s
            Q[2] = 0.25 * s
            Q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            Q[0] = (R[1, 0] - R[0, 1]) / s
            Q[1] = (R[0, 2] + R[2, 0]) / s
            Q[2] = (R[1, 2] + R[2, 1]) / s
            Q[3] = 0.25 * s

        return Q.tolist()

    def _initialize_database(self, image_paths, gt_poses, intrinsics):
        """
        Initialize the database using COLMAP's command-line interface, inserting camera and image information.

        Args:
            image_paths (list): List of processed image paths.
            gt_poses (list or np.ndarray): List of camera poses, each pose is a 4x4 matrix.
            intrinsics (list or np.ndarray): List of camera intrinsics, each intrinsic is a 3x3 matrix.
        """
        # Create a new COLMAP database
        print("Creating COLMAP database...")
        try:
            subprocess.run([
                self.colmap_executable,
                "database_creator",
                "--database_path", self.database_path
            ], check=True)
            print("COLMAP database created.")
        except subprocess.CalledProcessError as e:
            print(f"Database creation failed: {e}")
            return

        # Write cameras.txt and images.txt
        print("Writing cameras.txt and images.txt...")
        self._write_colmap_project(image_paths, gt_poses, intrinsics)
        print("cameras.txt and images.txt written.")
        # Copy the pre-made cameras and images to the project folder
        cameras_txt_path = os.path.join(self.project_path, "cameras.txt")
        images_txt_path = os.path.join(self.project_path, "images.txt")
        # Create sparse/0 folder
        os.makedirs(os.path.join(self.sparse_path, "0"), exist_ok=True)
        
        # Copy the cameras and images to the project folder
        shutil.copyfile(cameras_txt_path, os.path.join(self.sparse_path, "0", "cameras.txt"))
        shutil.copyfile(images_txt_path, os.path.join(self.sparse_path, "0", "images.txt"))
        
        # Create empty points3D.txt
        with open(os.path.join(self.sparse_path, "0", "points3D.txt"), 'w', encoding='utf-8') as f:
            # Write nothing
            pass


        # Import cameras.txt and images.txt into the database
        print("Importing cameras.txt and images.txt into COLMAP database...")
        try:
            # Note: COLMAP's command-line tools do not have a direct import function, so use pycolmap or manually insert
            # Here, pycolmap is used to manually insert camera and image information
            print(self.database_path)
            db = pycolmap.Database(self.database_path)

            # Insert camera information
            print("Inserting camera information into COLMAP database...")
            for cam in self.cameras:
                camera = pycolmap.Camera(
                    model=cam['model'],
                    width=cam['width'],
                    height=cam['height'],
                    params=cam['params']
                )
                camera_id = db.add_camera(camera)

            # Insert image information
            print("Inserting image information into COLMAP database...")
            for img in self.images_info:
                image = pycolmap.Image(
                    name=img['name'],
                    camera_id=img['camera_id'],
                    qvec=img['qvec'],
                    tvec=img['tvec']
                )
                image_id = db.add_image(image)

            db.commit()
            db.close()
            print("Import completed.")
        except Exception as e:
            print(f"Importing cameras and images failed: {e}")
            # Log traceback
            import traceback
            traceback.print_exc()
            return

    def _write_colmap_project(self, image_paths, gt_poses, intrinsics):
        """
        Write the cameras.txt and images.txt files required by COLMAP.

        Args:
            image_paths (list): List of processed image paths.
            gt_poses (list or np.ndarray): List of camera poses, each pose is a 4x4 matrix.
            intrinsics (list or np.ndarray): List of camera intrinsics, each intrinsic is a 3x3 matrix.
        """
        # Identify unique camera intrinsics and assign unique CAMERA_IDs
        print("Assigning unique CAMERA_IDs based on intrinsics...")
        intrin_to_camera_id = {}
        self.cameras = []
        self.images_info = []
        camera_id_counter = 1

        for intrin in intrinsics:
            # If it's a PyTorch tensor, convert to NumPy
            if isinstance(intrin, torch.Tensor):
                intrin = intrin.detach().cpu().numpy()
            # Ensure intrinsics are NumPy arrays
            intrin = np.array(intrin)
            # Create a key using floating-point numbers instead of tensors
            key = tuple(intrin.flatten().tolist())
            if key not in intrin_to_camera_id:
                intrin_to_camera_id[key] = camera_id_counter
                self.cameras.append({
                    'camera_id': camera_id_counter,
                    'model': 'PINHOLE',
                    'width': Image.open(image_paths[0]).width,  # Assuming same size
                    'height': Image.open(image_paths[0]).height,
                    'params': [intrin[0, 0], intrin[1, 1], intrin[0, 2], intrin[1, 2]]
                })
                camera_id_counter += 1

        # Write cameras.txt
        cameras_txt_path = os.path.join(self.project_path, "cameras.txt")
        with open(cameras_txt_path, 'w', encoding='utf-8') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
            for cam in self.cameras:
                f.write(f"{cam['camera_id']} {cam['model']} {cam['width']} {cam['height']} {' '.join(map(str, cam['params']))}\n")

        # Store camera information in memory for later database insertion
        # self.cameras already contains all camera information

        # Write images.txt
        images_txt_path = os.path.join(self.project_path, "images.txt")
        with open(images_txt_path, 'w', encoding='utf-8') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for idx, (img_path, pose, intrin) in enumerate(zip(image_paths, gt_poses, intrinsics), start=1):
                # Invert the pose
                cam2world = torch.linalg.inv(pose)
                qvec = self._rotation_matrix_to_quaternion(pose[:3, :3])
                tvec = pose[:3, 3]
                # If it's a PyTorch tensor, convert to NumPy
                if isinstance(intrin, torch.Tensor):
                    intrin = intrin.detach().cpu().numpy()
                intrin = np.array(intrin)
                key = tuple(intrin.flatten().tolist())
                camera_id = intrin_to_camera_id[key]
                image_name = os.path.relpath(img_path, self.cache_path).replace('\\', '/')
                image_name = image_name.split("/")[-1]

                self.images_info.append({
                    'image_id': idx,
                    'name': image_name,
                    'camera_id': camera_id,
                    'qvec': qvec,
                    'tvec': tvec.tolist()
                })

                f.write(f"{idx} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {image_name}\n")
                f.write("\n")  # POINTS2D[] line left empty

    def _run_feature_extractor(self):
        """
        Invoke COLMAP via command line to perform feature extraction.
        """
        try:
            subprocess.run([
                self.colmap_executable,
                "feature_extractor",
                "--database_path", self.database_path,
                "--image_path", self.image_dir,
                "--SiftExtraction.use_gpu", "1"
            ], check=True)
            print("Feature extraction completed.")
        except subprocess.CalledProcessError as e:
            print(f"Feature extraction failed: {e}")
            raise

    def _run_exhaustive_matcher(self):
        """
        Invoke COLMAP via command line to perform feature matching.
        """
        try:
            subprocess.run([
                self.colmap_executable,
                "exhaustive_matcher",
                "--database_path", self.database_path,
                "--SiftMatching.use_gpu", "1"
            ], check=True)
            print("Feature matching completed.")
        except subprocess.CalledProcessError as e:
            print(f"Feature matching failed: {e}")
            raise

    def _run_colmap_reconstruction(self):
        """
        Invoke COLMAP via command line to perform sparse reconstruction.
        """
        
        try:
            subprocess.run([
                self.colmap_executable,
                "point_triangulator",
                "--database_path", self.database_path,
                "--image_path", self.image_dir,
                "--input_path", os.path.join(self.sparse_path, "0"),
                "--output_path", os.path.join(self.sparse_path, "0"),
                "--log_to_stderr", "1"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Point triangulation failed: {e}")
            raise

    def _export_point_cloud(self):
        """
        Export the sparse point cloud reconstructed by COLMAP to PLY format.
        """
        # Assume the reconstructed model is saved in the sparse/0 directory
        model_path = os.path.join(self.sparse_path, "0")
        ply_output_path = os.path.join(self.cache_path, "reconstruction.ply")

        if not os.path.exists(model_path):
            print(f"Reconstruction model not found at {model_path}.")
            return

        # Export point cloud to PLY format
        try:
            subprocess.run([
                self.colmap_executable,
                "model_converter",
                "--input_path", model_path,
                "--output_path", ply_output_path,
                "--output_type", "PLY"
            ], check=True)
            print(f"Point cloud exported to {ply_output_path}")

            # Optionally visualize using trimesh
            # self.visualize_ply(ply_output_path)
            return ply_output_path

        except subprocess.CalledProcessError as e:
            print(f"Point cloud export failed: {e}")
            raise

    def _check_database(self, table_name):
        """
        Check the number of entries in the specified table of the database.

        Args:
            table_name (str): Name of the table to check, such as 'keypoints', 'descriptors', 'matches'.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"Database check - {table_name}: {count} entries.")
            conn.close()
        except Exception as e:
            print(f"Database check failed for table {table_name}: {e}")

    def visualize_ply(self, ply_path):
        """
        Visualize the PLY point cloud using trimesh.

        Args:
            ply_path (str): Path to the PLY file.
        """
        try:
            point_cloud = trimesh.load(ply_path)
            if isinstance(point_cloud, trimesh.Scene):
                # If loaded as a Scene, merge all geometries
                point_cloud = trimesh.util.concatenate(point_cloud.dump())
            point_cloud.show()
        except Exception as e:
            print(f"Failed to visualize PLY file: {e}")