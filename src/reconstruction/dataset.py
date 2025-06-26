"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 21:25:43
Description:

"""
# dataset.py provide api for benchmark dataset's loading and processing
# also provide custom dataset for evaluation

import os
import torch
import sys
import json
import yaml
import argparse
from src.datasets.base import BoxDreamerBaseDataset
from src.datasets.linemod import LINEMOD_BoxDreamer
from src.datasets.co3d import Co3DV2Dataset
from src.datasets import make_dataloader
from src.datasets.onepose import OnePoseDataset
from src.datasets.objaverse import ObjaverseBoxDreamerDataset
from src.datasets.linemodo import LINEMOD_Occulusion
from src.datasets.ycbv import YCBV_BoxDreamer
from src.datasets.moped import MOPED_BoxDreamer


def get_dataset(cfgs, dataset_name):
    if dataset_name == "LINEMOD":
        return LINEMOD_BoxDreamer(cfgs.LINEMOD.config, split="test")
    elif dataset_name == "CO3D":
        return Co3DV2Dataset(cfgs.CO3D.config, split="test")
    elif dataset_name == "OnePose":
        return OnePoseDataset(cfgs.OnePose.config, split="test")
    elif dataset_name == "OnePose_Lowtexture":
        return OnePoseDataset(cfgs.OnePose_Lowtexture.config, split="test")
    elif dataset_name == "Objaverse":
        return ObjaverseBoxDreamerDataset(cfgs.Objaverse.config, split="test")
    elif dataset_name == "LINEMODO":
        return LINEMOD_Occulusion(cfgs.LINEMODO.config, split="test")
    elif dataset_name == "YCBV":
        return YCBV_BoxDreamer(cfgs.YCBV.config, split="test")
    elif dataset_name == "MOPED":
        return MOPED_BoxDreamer(cfgs.MOPED.config, split="test")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def get_dl(cfgs, dataset):
    test_loader_params = {
        "batch_size": 1,  # do not change this
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": True,
        "drop_last": False,
    }
    test_dataset = get_dataset(cfgs, dataset)
    test_loader = make_dataloader(test_dataset, test_loader_params)

    return test_loader
