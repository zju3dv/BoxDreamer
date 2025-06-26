import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
from src.utils.camera_transform import *
from src.datasets.base import (
    _crop_image,
    calculate_crop_parameters,
    pad_and_resize_image,
    adjust_intrinsic_matrix,
    random_crop_and_scale,
    square_bbox,
    BoxDreamerBaseDataset,
    DynamicBatchSampler,
    make_dataloader,
)
from torch.utils.data import SequentialSampler


def array_to_tensor(array, dtype):
    return torch.tensor(array, dtype=dtype)


# Utility function to create a dummy image
def create_dummy_image(width=100, height=100, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color)


# Test calculate_crop_parameters
def test_calculate_crop_parameters():
    image = create_dummy_image(200, 100)
    bbox = np.array([50, 25, 150, 75])
    crop_dim = 100
    img_size = 224

    crop_params = calculate_crop_parameters(image, bbox, crop_dim, img_size)
    assert isinstance(crop_params, torch.Tensor)
    assert crop_params.shape == torch.Size([4])


# Test _crop_image
def test_crop_image():
    image = create_dummy_image(200, 100)
    bbox = np.array([50, 25, 150, 75])

    cropped_image = _crop_image(image, bbox)
    assert isinstance(cropped_image, Image.Image)
    assert cropped_image.size == (100, 50)

    # Test with white background padding
    bbox_outside = np.array([-10, -10, 210, 110])
    cropped_image_padded = _crop_image(image, bbox_outside, white_bg=True)
    assert isinstance(cropped_image_padded, Image.Image)
    assert cropped_image_padded.size == (200, 100)


# Test adjust_intrinsic_matrix
def test_adjust_intrinsic_matrix():
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    scale = (1.5, 1.5)
    crop_offset = (50, 30)

    K_new = adjust_intrinsic_matrix(K, scale, crop_offset)
    expected_K_new = K.copy()
    expected_K_new[0, 0] *= 1.5
    expected_K_new[1, 1] *= 1.5
    expected_K_new[0, 2] = expected_K_new[0, 2] * 1.5 - 50 * 1.5
    expected_K_new[1, 2] = expected_K_new[1, 2] * 1.5 - 30 * 1.5

    np.testing.assert_array_almost_equal(K_new, expected_K_new)


# Test square_bbox
def test_square_bbox():
    bbox = np.array([50, 25, 150, 75])
    square = square_bbox(bbox, padding=0.1)
    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    expected_side = side * 1.1
    center = np.array([100, 50])
    expected_bbox = np.array(
        [
            center[0] - expected_side / 2,
            center[1] - expected_side / 2,
            center[0] + expected_side / 2,
            center[1] + expected_side / 2,
        ]
    )
    square = square.astype(np.int32)
    expected_bbox = expected_bbox.astype(np.int32)
    np.testing.assert_array_almost_equal(square, expected_bbox)


# Test pad_and_resize_image
def test_pad_and_resize_image():
    image = create_dummy_image(200, 100)
    bbox = np.array([50, 25, 150, 75])
    img_size = [224, 224]

    transformed_image, mask, crop_paras, max_sq_bbox = pad_and_resize_image(
        image=image,
        crop_longest=False,
        img_size=img_size,
        mask=None,
        bbox_anno=bbox,
        transform=None,
    )

    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape[0] == 3  # RGB channels
    assert transformed_image.shape[1] == img_size[0]
    assert transformed_image.shape[2] == img_size[1]


# Test random_crop_and_scale
def test_random_crop_and_scale():
    image = create_dummy_image(200, 100)
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    bbox = np.array([50, 25, 150, 75])

    with patch("random.uniform", return_value=1.0), patch(
        "random.randint", return_value=0
    ):
        transformed_image, K_new, bbox_cropped = random_crop_and_scale(
            image=image,
            K=K,
            bbox=bbox,
            scale_range=(0.8, 1.2),
            crop_scale_range=(0.5, 1.0),
            aspect_ratio_range=(0.75, 1.33),
        )

    assert isinstance(transformed_image, Image.Image)
    assert isinstance(K_new, np.ndarray)
    assert K_new.shape == (3, 3)
    assert isinstance(bbox_cropped, np.ndarray)
    assert bbox_cropped.shape == (4,)


# Test reproj_pytorch
def test_reproj_pytorch(mock_dataset: BoxDreamerBaseDataset):
    K = torch.tensor([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)
    pts_3d = torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [-1.0, -1.0, 5.0]])

    reproj_2d = mock_dataset.reproj_pytorch(K, pose, pts_3d)
    expected = torch.tensor([[320.0, 240.0], [520.0, 440.0], [120.0, 40.0]])

    torch.testing.assert_close(reproj_2d, expected, atol=1e-5, rtol=1e-5)


# Mock BoxDreamerBaseDataset's dependencies
@pytest.fixture
def mock_dataset_config():
    config = MagicMock()
    config.image_size = [224, 224]
    config.length = 5
    config.dynamic_length = False
    config.stride = 1
    config.dynamic_stride = False
    config.random_stride = False
    config.augmentation_method = None
    config.pose_augmentation = False
    config.intri_augmentation = False
    config.compute_optical = False
    config.max_norm = 1.0
    config.precision = "32"
    return config


@pytest.fixture
def mock_dataset(mock_dataset_config) -> BoxDreamerBaseDataset:
    dataset = BoxDreamerBaseDataset(config=mock_dataset_config, split="train")
    # Mock the data storage
    dataset.images = {
        "train": {"cat1": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]}
    }
    dataset.boxes = {
        "train": {"cat1": [np.array([50, 50, 150, 150]) for _ in range(5)]}
    }
    dataset.poses = {"train": {"cat1": [np.eye(4) for _ in range(5)]}}
    dataset.intrinsics = {
        "train": {
            "cat1": [
                np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]) for _ in range(5)
            ]
        }
    }
    dataset.model_paths = {"cat1": "path/to/model.obj"}
    dataset.cat_len = {"train": {"cat1": 5}, "ref": {"cat1": 5}}
    dataset.dataset = "default_dataset"
    dataset.dynamic_length = False
    return dataset


# Test BoxDreamerBaseDataset.__len__
def test_dataset_length(mock_dataset):
    assert len(mock_dataset) == 5


# Test BoxDreamerBaseDataset.read_images
def test_read_images(mock_dataset):
    with patch(
        "cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch(
        "cv2.cvtColor", return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch(
        "cv2.resize", return_value=np.zeros((100, 100, 3), dtype=np.float32)
    ):
        images = mock_dataset.read_images(["img1.jpg", "img2.jpg"])
        assert len(images) == 2
        assert images[0].shape == (100, 100, 3)
        assert images[0].dtype == np.float32


# Test BoxDreamerBaseDataset.read_images_pil
def test_read_images_pil(mock_dataset):
    with patch("PIL.Image.open", return_value=create_dummy_image()):
        images = mock_dataset.read_images_pil(["img1.jpg", "img2.jpg"])
        assert len(images) == 2
        assert isinstance(images[0], Image.Image)


# Test BoxDreamerBaseDataset.read_boxes
def test_read_boxes(mock_dataset):
    with patch.object(
        mock_dataset, "extract_bboxes", return_value=np.array([50, 50, 150, 150])
    ):
        boxes = mock_dataset.read_boxes(["mask.png"])
        assert len(boxes) == 1
        np.testing.assert_array_equal(boxes[0], np.array([50, 50, 150, 150]))


# Test BoxDreamerBaseDataset.extract_bboxes
def test_extract_bboxes(mock_dataset):
    mock_mask = np.zeros((200, 200), dtype=np.uint8)
    mock_mask[50:150, 50:150] = 255
    with patch("cv2.imread", return_value=mock_mask):
        bbox = mock_dataset.extract_bboxes("mask.png")
        np.testing.assert_array_equal(bbox, np.array([50, 50, 150, 150]))


# Test DynamicBatchSampler
def test_dynamic_batch_sampler(mock_dataset_config, mock_dataset):
    sampler = SequentialSampler(mock_dataset)
    batch_size = 2
    drop_last = False
    batch_sampler = DynamicBatchSampler(sampler, batch_size, drop_last, mock_dataset)

    batches = list(batch_sampler)
    assert (
        len(batches) == 3
    )  # 5 samples with batch size 2 => 3 batches (last one with 1 sample)
    assert all(len(batch) == 2 for batch in batches[:-1])
    assert len(batches[-1]) == 1


def test_make_dataloader(mock_dataset_config, mock_dataset):
    import omegaconf

    with patch("src.datasets.base.DataLoader") as mock_dataloader:
        cfg = omegaconf.OmegaConf.create(
            {
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": False,
                "drop_last": False,
                "pin_memory": False,
            }
        )

        dataloader = make_dataloader(mock_dataset, cfg)

        mock_dataloader.assert_called_once()

        args, kwargs = mock_dataloader.call_args

        assert args[0] == mock_dataset
        assert kwargs["num_workers"] == 0
        assert kwargs["pin_memory"] == False

    assert dataloader == mock_dataloader.return_value


def test_make_dataloader_dynamic(mock_dataset_config, mock_dataset):
    import omegaconf

    with patch("src.datasets.base.DataLoader") as mock_dataloader:
        cfg = omegaconf.OmegaConf.create(
            {
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": False,
                "drop_last": False,
                "pin_memory": False,
            }
        )
        mock_dataset.dynamic_length = True

        dataloader = make_dataloader(mock_dataset, cfg)

        mock_dataloader.assert_called_once()

        args, kwargs = mock_dataloader.call_args

        assert args[0] == mock_dataset
        assert "batch_sampler" in kwargs
        assert kwargs["num_workers"] == 0
        assert kwargs["pin_memory"] == False

    assert dataloader == mock_dataloader.return_value
