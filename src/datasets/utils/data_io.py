"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:09:39
Description:

"""
import pickle
import h5py
import numpy as np
import torch


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def load_h5(file_path, transform_slash=True, parallel=False):
    """Load the whole h5 file into memory (not memmaped)"""
    with h5py.File(file_path, "r") as f:
        # if parallel:
        #     Parallel()
        data = {
            k if not transform_slash else k.replace("+", "/"): v.__array__()
            for k, v in f.items()
        }
    return data


def save_h5(dict_to_save, filename, transform_slash=True):
    """Saves dictionary to hdf5 file."""
    with h5py.File(filename, "w") as f:
        for (
            key
        ) in (
            dict_to_save
        ):  # h5py doesn't allow '/' in object name (will leads to sub-group)
            f.create_dataset(
                key.replace("/", "+") if transform_slash else key,
                data=dict_to_save[key],
            )


def process_resize(w, h, resize, df=None):
    if resize is not None:
        assert len(resize) > 0 and len(resize) <= 2
        if len(resize) == 1 and resize[0] > -1:  # resize the larger side
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            mask[: inp.shape[0], : inp.shape[1]] = 1
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            mask[:, : inp.shape[1], : inp.shape[2]] = 1
    else:
        raise NotImplementedError()
    return padded, mask


def grayscale2tensor(image, mask=None):
    return torch.from_numpy(image / 255.0).float()[None]  # (1, h, w)


def mask2tensor(mask):
    return torch.from_numpy(mask).float()  # (h, w)


def txt2tensor(path):
    data = np.loadtxt(path)
    return torch.from_numpy(data).float()


def npy2tensor(path):
    data = np.load(path)
    return torch.from_numpy(data).float()


def npz2tensor(path, key):
    data = np.load(path)
    return torch.from_numpy(data[key]).float()
