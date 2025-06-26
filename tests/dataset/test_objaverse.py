import unittest
import os
from omegaconf import OmegaConf
from src.datasets.objaverse import ObjaverseBoxDreamerDataset
import numpy as np
import json
from unittest.mock import patch, mock_open


class TestObjaverseBoxDreamerDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary config for testing
        self.config = OmegaConf.create(
            {
                "base": {
                    "image_size": 224,
                    "length": 6,
                    "dynamic_length": False,
                    "stride": [5, 25],
                    "dynamic_stride": True,
                    "random_stride": False,
                    "augmentation_method": None,
                    "pose_augmentation": False,
                    "intri_augmentation": False,
                    "compute_optical": False,
                    "max_norm": False,
                    "precision": "float32",
                },
                "subdir_num": 2,
                "root": os.path.join(os.getcwd(), "data/objaverse"),
            }
        )
        self.split = "train"
        self.dataset = ObjaverseBoxDreamerDataset(self.config, self.split)

    def test_initialization(self):
        self.assertEqual(self.dataset.dataset, "objaverse")
        self.assertEqual(self.dataset.split, self.split)
        self.assertEqual(self.dataset.root, self.config.root)

    def test_load_data(self):
        self.dataset.load_data()
        self.assertIn("train", self.dataset.images)
        self.assertIn("ref", self.dataset.images)
        self.assertIn("train", self.dataset.boxes)
        self.assertIn("ref", self.dataset.boxes)
        self.assertIn("train", self.dataset.poses)
        self.assertIn("ref", self.dataset.poses)
        self.assertIn("train", self.dataset.intrinsics)
        self.assertIn("ref", self.dataset.intrinsics)
        self.assertIn("train", self.dataset.cat_len)
        self.assertIn("ref", self.dataset.cat_len)

    def test_load_data_from_dir(self):
        # objaverse have no test split
        # check whether the test split is loaded or not
        try:
            self.dataset.load_data("test")
            raise Exception("Test split is loaded")
        except:
            pass

    def test_all_file_sorted(self):
        self.dataset.load_data()
        for key in self.dataset.images["train"]:
            self.assertEqual(
                self.dataset.images["train"][key],
                sorted(self.dataset.images["train"][key]),
            )
        for key in self.dataset.poses["train"]:
            self.assertEqual(
                self.dataset.poses["train"][key],
                sorted(self.dataset.poses["train"][key]),
            )
        for key in self.dataset.intrinsics["train"]:
            self.assertEqual(
                self.dataset.intrinsics["train"][key],
                sorted(self.dataset.intrinsics["train"][key]),
            )
        for key in self.dataset.cat_len["train"]:
            self.assertEqual(
                self.dataset.cat_len["train"][key],
                len(self.dataset.images["train"][key]),
            )

    def all_file_have_same_len(self):
        self.dataset.load_data()
        for key in self.dataset.images["train"]:
            self.assertEqual(
                len(self.dataset.images["train"][key]),
                len(self.dataset.poses["train"][key]),
            )
            self.assertEqual(
                len(self.dataset.images["train"][key]),
                len(self.dataset.intrinsics["train"][key]),
            )
            self.assertEqual(
                len(self.dataset.images["train"][key]),
                self.dataset.cat_len["train"][key],
            )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"frames": [{"w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "fx": 500, "fy": 500, "cx": 320, "cy": 240}]}',
    )
    def test_read_poses(self, mock_file):
        pose_paths = ["dummy_path.json"]
        poses = self.dataset.read_poses(pose_paths, [0])
        expected_poses = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])]
        np.testing.assert_array_equal(poses[0], expected_poses[0])

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"frames": [{"w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "fx": 500, "fy": 500, "cx": 320, "cy": 240}, {"w2c": [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], "fx": 500, "fy": 500, "cx": 320, "cy": 240}]}',
    )
    def test_read_multiple_poses(self, mock_file):
        pose_paths = ["dummy_path.json"]
        poses = self.dataset.read_poses(pose_paths, [1, 0])
        expected_poses = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]),
        ]
        np.testing.assert_array_equal(poses[1], expected_poses[0])
        np.testing.assert_array_equal(poses[0], expected_poses[1])

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"frames": [{"w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "fx": 500, "fy": 500, "cx": 320, "cy": 240}]}',
    )
    def test_read_intrinsics(self, mock_file):
        intri_paths = ["dummy_path.json"]
        intrinsics = self.dataset.read_intrinsics(intri_paths, [0])
        expected_intrinsics = [np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])]
        np.testing.assert_array_equal(intrinsics[0], expected_intrinsics[0])


if __name__ == "__main__":
    unittest.main()
