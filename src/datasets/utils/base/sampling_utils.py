"""Utility functions for sampling strategies."""

import os
import json
import numpy as np
from typing import Dict, Any, List


def select_ref_indices(
    random_stride: bool,
    fps_sampling: bool,
    uniform_sampling: bool,
    stride: int,
    dynamic_stride: bool,
    min_stride: int,
    max_stride: int,
    ref_length: int,
    max_ref_idx: int,
    images: Dict[str, Any],
    query_cat: str,
    query_q_idx: int,
    dataset: str,
) -> np.ndarray:
    """Select reference indices based on specified sampling strategies. Falls
    back to uniform sampling if the chosen method fails.

    Args:
        random_stride (bool): Use random stride sampling if True.
        fps_sampling (bool): Use Farthest Point Sampling (FPS) if True.
        uniform_sampling (bool): Use uniform sampling if True.
        stride (int): Fixed stride value for stride-based sampling.
        dynamic_stride (bool): Use dynamic stride if True.
        min_stride (int): Minimum stride value for dynamic stride sampling.
        max_stride (int): Maximum stride value for dynamic stride sampling.
        ref_length (int): Number of reference indices to sample.
        max_ref_idx (int): Maximum reference index (inclusive).
        images (Dict[str, Any]): Dictionary containing image information.
        query_cat (str): Category of the query.
        query_q_idx (int): Query index used for FPS sampling.
        dataset (str): Name of the dataset (e.g., 'linemod').

    Returns:
        np.ndarray: Array of selected reference indices.
    """
    try:
        if random_stride:
            # Randomly select `ref_length` unique indices
            if ref_length > max_ref_idx + 1:
                raise ValueError(
                    f"ref_length {ref_length} exceeds the number of available indices {max_ref_idx + 1}."
                )
            ref_idxs = np.random.choice(max_ref_idx + 1, size=ref_length, replace=False)

        elif fps_sampling:
            # has been discarded in the latest version of the code
            raise NotImplementedError(
                "Farthest Point Sampling (FPS) is not supported in the latest version, because this process is now \
                    handled by the preprocessing script."
            )

        elif uniform_sampling:
            if ref_length <= 0:
                raise ValueError("ref_length must be positive for uniform sampling.")

            if max_ref_idx + 1 == ref_length:
                ref_idxs = np.arange(0, max_ref_idx + 1)
            else:
                try:
                    stride_value = max_ref_idx // ref_length if ref_length > 0 else 1
                    ref_idxs = np.arange(0, max_ref_idx + 1, stride_value)
                except ZeroDivisionError:
                    # this is caused by dataset length < ref_length
                    raise ValueError(
                        f"Dataset length {max_ref_idx + 1} is less than ref_length {ref_length}."
                    )
                    print(max_ref_idx, ref_length)
                    ref_idxs = np.arange(0, max_ref_idx + 1)

            if len(ref_idxs) > ref_length:
                ref_idxs = ref_idxs[:ref_length]
            elif len(ref_idxs) < ref_length:
                last_idx = ref_idxs[-1] if len(ref_idxs) > 0 else 0
                padding = np.full(ref_length - len(ref_idxs), last_idx)
                ref_idxs = np.concatenate((ref_idxs, padding))

        elif not random_stride:
            # Stride-based sampling
            if dynamic_stride:
                if min_stride > max_stride:
                    raise ValueError("min_stride cannot be greater than max_stride.")
                stride_value = np.random.randint(min_stride, max_stride + 1)
            else:
                stride_value = stride

            if stride_value <= 0:
                raise ValueError("Stride must be positive.")

            # Calculate starting index
            max_start_idx = max_ref_idx - (stride_value * ref_length)
            if max_start_idx < 0:
                stride_value = (
                    max(max_ref_idx // ref_length, 1) if ref_length > 0 else 1
                )
                max_start_idx = max_ref_idx - (stride_value * ref_length)
                max_start_idx = max(max_start_idx, 0)

            if dataset.lower() == "linemod":
                start_idx = (
                    np.random.randint(0, max_start_idx + 1) if max_start_idx > 0 else 0
                )
            else:
                start_idx = query_q_idx + stride_value
                start_idx = min(start_idx, max_start_idx)

            ref_idxs = np.arange(
                start_idx, start_idx + stride_value * ref_length, stride_value
            )

            # Adjust indices if they exceed maximum reference index
            if ref_idxs[-1] > max_ref_idx:
                start_idx = max_ref_idx - stride_value * ref_length
                start_idx = max(start_idx, 0)
                ref_idxs = np.arange(
                    start_idx, start_idx + stride_value * ref_length, stride_value
                )

            # Ensure the correct number of indices
            if len(ref_idxs) != ref_length:
                if len(ref_idxs) > ref_length:
                    ref_idxs = ref_idxs[:ref_length]
                else:
                    additional_needed = ref_length - len(ref_idxs)
                    additional_start = max(
                        start_idx - stride_value * additional_needed, 0
                    )
                    additional_idxs = np.arange(
                        additional_start, start_idx, stride_value
                    )
                    additional_idxs = np.clip(additional_idxs, 0, max_ref_idx)
                    ref_idxs = np.concatenate((additional_idxs, ref_idxs))
                    if len(ref_idxs) < ref_length:
                        last_idx = ref_idxs[-1] if len(ref_idxs) > 0 else 0
                        padding = np.full(ref_length - len(ref_idxs), last_idx)
                        ref_idxs = np.concatenate((ref_idxs, padding))
                    ref_idxs = ref_idxs[:ref_length]

            ref_idxs = np.sort(ref_idxs).astype(int)

        else:
            raise ValueError("No valid sampling method selected.")

    except Exception as e:
        # Fallback to uniform sampling in case of any errors
        print(
            f"Sampling method failed with error: {e}. Falling back to uniform sampling."
        )
        if max_ref_idx + 1 < ref_length:
            raise ValueError(
                f"Reference database length {max_ref_idx + 1} is less than ref_length {ref_length}."
            )
        stride_value = max_ref_idx // ref_length if ref_length > 0 else 1
        ref_idxs = np.arange(0, max_ref_idx + 1, stride_value)

        if len(ref_idxs) > ref_length:
            ref_idxs = ref_idxs[:ref_length]
        elif len(ref_idxs) < ref_length:
            last_idx = ref_idxs[-1] if len(ref_idxs) > 0 else 0
            padding = np.full(ref_length - len(ref_idxs), last_idx)
            ref_idxs = np.concatenate((ref_idxs, padding))

    return ref_idxs.astype(int)
