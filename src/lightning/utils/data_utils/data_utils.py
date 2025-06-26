"""
Author: Yuanhong Yu
Date: 2025-03-17 16:20:11
LastEditTime: 2025-03-17 16:20:19
Description:

"""
import torch
import numpy as np
from itertools import chain
import torch.distributed as dist
from src.utils.comm import gather


class DataProcessor:
    """Utility class for data processing operations."""

    @staticmethod
    def back_to_cpu(data):
        """Recursively move tensors to CPU and convert half precision to
        float32.

        Args:
            data: Input data which can be a tensor, dict, or list

        Returns:
            The same data structure with tensors moved to CPU
        """
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = DataProcessor.back_to_cpu(v)
        elif isinstance(data, list):
            return [DataProcessor.back_to_cpu(item) for item in data]
        elif isinstance(data, torch.Tensor):
            dtype = data.dtype
            if dtype == torch.float16 or dtype == torch.bfloat16:
                return data.detach().cpu().to(torch.float32)
            return data.detach().cpu()
        return data

    @staticmethod
    def detach(data):
        """Recursively detach tensors from computation graph.

        Args:
            data: Input data which can be a tensor, dict, or list

        Returns:
            The same data structure with tensors detached from computation graph
        """
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = DataProcessor.detach(v)
        elif isinstance(data, list):
            return [DataProcessor.detach(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.detach()
        return data

    @staticmethod
    def flatten_data(data):
        """Flatten a list of data into a single data structure.

        Args:
            data: A list of data objects (dict, np.array, etc.)

        Returns:
            Flattened data structure
        """
        # data is a list of np.array / dict / or any other picklable object
        # for each type of data, accumulate them into a org picklable object
        assert isinstance(data, list), "data should be a list of picklable objects"
        if len(data) == 0:
            return data
        if isinstance(data[0], dict):
            ret = {}
            for k in data[0].keys():
                ret[k] = DataProcessor.flatten_data([d[k] for d in data])
            return ret
        elif isinstance(data[0], np.ndarray):
            return np.concatenate(data, axis=0)
        elif isinstance(data[0], list):
            return list(chain(*data))
        else:
            return data

    @staticmethod
    def gather_data(data, tgt):
        """Gather data from all processes and flatten it.

        Args:
            data: Data to gather
            tgt: Target process to gather data to

        Returns:
            Gathered and flattened data
        """
        return DataProcessor.flatten_data(gather(data, tgt))
