# load pytorch lightning ckpt and only save state_dict

import torch
import os
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="path to pytorch lightning ckpt")
    # default save to the same directory
    parser.add_argument(
        "--save_dir", type=str, default=None, help="path to save state_dict"
    )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    state_dict = ckpt["state_dict"]
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.dirname(args.ckpt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir, args.ckpt.split("/")[-1].replace(".ckpt", "_clean.ckpt")
    )

    # keep 'state_dict' only
    new_ckpt = {"state_dict": state_dict}
    torch.save(new_ckpt, save_path)

    print(f"state_dict saved to {save_path}")
