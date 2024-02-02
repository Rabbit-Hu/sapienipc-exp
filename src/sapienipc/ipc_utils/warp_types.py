from typing import Union
import numpy as np
import torch

import warp as wp
from warp.types import type_size_in_bytes


mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)
mat22 = wp.types.matrix(shape=(2, 2), dtype=wp.float32)


def wp_slice(a: wp.array, start, end):
    """Utility function to slice a warp array along the first dimension
    """

    assert a.is_contiguous
    assert 0 <= start <= end <= a.shape[0]
    return wp.array(
        ptr=a.ptr + start * a.strides[0],
        dtype=a.dtype,
        shape=(end - start, *a.shape[1:]),
        strides=a.strides,
        device=a.device,
        copy=False,
        owner=False,
    )


def convert_to_wp_array(
    a: Union[np.ndarray, torch.Tensor, wp.array, list], dtype=None, device=None
):
    """Utility function to convert numpy or torch tensor to warp array
    """
    if isinstance(a, np.ndarray):
        return wp.from_numpy(a, dtype=dtype, device=device)
    elif isinstance(a, torch.Tensor):
        return wp.from_torch(a, dtype=dtype).to(device)
    elif isinstance(a, list):
        return wp.from_numpy(np.array(a), dtype=dtype, device=device)
    elif isinstance(a, wp.array):
        assert dtype is None or dtype == a.dtype
        assert device is None or device == a.device
        return a
    else:
        raise ValueError(f"unsupported type {type(a)}")


def torch_wait_wp_stream(self, device=None):
    torch_stream = wp.stream_from_torch(torch.cuda.current_stream())
    warp_stream = wp.get_stream(device)
    torch_stream.wait_stream(warp_stream)


def wp_wait_torch_stream(self, device=None):
    torch_stream = wp.stream_from_torch(torch.cuda.current_stream())
    warp_stream = wp.get_stream(device)
    warp_stream.wait_stream(torch_stream)
