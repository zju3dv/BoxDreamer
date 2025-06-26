import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.datasets.onepose import OnePoseDataset
from src.datasets.co3d import Co3DV2Dataset
from src.datasets.linemod import LINEMOD_BoxDreamer
import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import psutil


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze pose data from various datasets."
    )

    # Dataset related parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["onepose", "co3d", "linemod"],
        default="OnePose",
        help="Choose the dataset to analyze",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Choose the dataset split (train or test)",
    )

    # DataLoader parameters
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for DataLoader",
    )

    # Dataset configuration parameters
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size of the images"
    )
    parser.add_argument("--length", type=int, default=6, help="Length of the sequence")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride of the sequence, can input multiple values",
    )
    parser.add_argument(
        "--coordinate",
        type=str,
        default="first_camera",
        help="Coordinate system setting (e.g., first_camera, object)",
    )

    # Other optional parameters can be added as needed
    # ...

    return parser.parse_args()


def get_dataset_config(args):
    # Define root directories for each dataset
    dataset_roots = {"onepose": "data/onepose", "co3d": "data", "linemod": "data/lm"}

    # Base configuration
    base_config = {
        "image_size": args.image_size,
        "length": args.length,
        "dynamic_length": False,
        "stride": args.stride,
        "dynamic_stride": False,
        "random_stride": False,
        "uniform_sampling": False,
        "augmentation_method": None,
        "pose_augmentation": False,
        "intri_augmentation": False,
        "compute_optical": True,
        "max_norm": False,
        "precision": "32",
        "coordinate": args.coordinate,
        "use_bbox": False,
        "use_mask": False,
    }

    config = {
        "name": args.dataset,
        "config": {
            "base": base_config,
            "root": os.path.join(os.getcwd(), dataset_roots[args.dataset]),
        },
    }

    return OmegaConf.create(config)


def print_memory_info():
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")


def compute_angle(rotation_matrix):
    trace = rotation_matrix.trace()
    return np.arccos(np.clip((trace - 1) / 2, -1, 1))


def main():
    args = parse_arguments()
    print("Parsed command line arguments:", args)

    # Print memory information
    print_memory_info()

    # Get dataset configuration
    dataset_config = get_dataset_config(args)
    print("Using dataset configuration:")
    print(dataset_config)

    if args.dataset == "onepose":
        dataset = OnePoseDataset(dataset_config["config"], args.split)
    elif args.dataset == "co3d":
        dataset = Co3DV2Dataset(dataset_config["config"], args.split)
    elif args.dataset == "linemod":
        dataset = LINEMOD_BoxDreamer(dataset_config["config"], args.split)

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    # Set output path
    dump_path = f"/home/yyh/lab/BoxDreamer/statistic/{args.dataset.lower()}-{args.coordinate}--{args.split}--{args.length}--{args.stride}"
    os.makedirs(dump_path, exist_ok=True)

    angles = []
    translations = []

    try:
        for data in tqdm.tqdm(dataloader, desc="pose analysis", total=len(dataloader)):
            try:
                query_pose = data["poses"][0][-1].numpy()
                scaled_query_pose = data["original_poses"][0][-1].numpy()

                angles.append(np.rad2deg(compute_angle(query_pose[:3, :3])))
                translations.append(query_pose[:3, 3])
            except Exception as e:
                print(f"Error processing data: {e}")
                continue
    except Exception as e:
        print(f"Error iterating through DataLoader: {e}")

    # Convert to NumPy arrays
    angles = np.array(angles)
    translations = np.array(translations)

    # Plot angle distribution histogram
    plt.hist(angles, bins=100)
    plt.title("Angle Distribution")
    plt.xlabel("Angle (degree)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(dump_path, "angles.png"))
    plt.close()

    # Print angle distribution statistics
    angles_stats = {
        "min": np.min(angles),
        "max": np.max(angles),
        "mean": np.mean(angles),
        "std": np.std(angles),
    }
    print("Angle distribution statistics: ", angles_stats)

    # Save angle distribution statistics as JSON file
    with open(os.path.join(dump_path, "angles.json"), "w") as f:
        json.dump(angles_stats, f)

    # Calculate mean and standard deviation of translation vectors
    translations_stats = {
        "mean": np.mean(translations, axis=0).tolist(),
        "std": np.std(translations, axis=0).tolist(),
    }
    print("Translation vector statistics: ", translations_stats)

    # Save translation vector statistics as JSON file
    with open(os.path.join(dump_path, "translation.json"), "w") as f:
        json.dump(translations_stats, f)


if __name__ == "__main__":
    main()
