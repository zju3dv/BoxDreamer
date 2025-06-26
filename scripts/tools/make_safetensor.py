# load torch ckpt and create huggingface safetensor

import torch
from safetensors.torch import save_file
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)

    args = parser.parse_args()
    ckpt = torch.load(args.ckpt)["state_dict"]
    ext = args.ckpt.split(".")[-1]

    safetensor_path = args.ckpt.replace(f".{ext}", ".safetensor")
    save_file(ckpt, safetensor_path)
    print(f"save safetensor to {safetensor_path}")
