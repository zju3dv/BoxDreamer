"""
Author: Yuanhong Yu
Date: 2025-03-17 16:21:12
LastEditTime: 2025-03-17 16:21:21
Description:

"""
import gc
import torch
from contextlib import contextmanager


class MemoryManager:
    """Utility class for memory management."""

    @staticmethod
    def clean_variables(*variables):
        """Delete variables and clean CUDA cache.

        Args:
            *variables: Variables to be deleted
        """
        for var in variables:
            # if var is dict, del it recursively
            if isinstance(var, dict):
                for k, v in var.items():
                    MemoryManager.clean_variables(v)
            else:
                del var
        torch.cuda.empty_cache()

    @staticmethod
    @contextmanager
    def auto_cleanup():
        """Context manager for automatic memory cleanup.

        Example usage:
        with MemoryManager.auto_cleanup():
            # Code that might create a lot of temporary variables
        """
        try:
            yield
        finally:
            gc.collect()
            torch.cuda.empty_cache()
