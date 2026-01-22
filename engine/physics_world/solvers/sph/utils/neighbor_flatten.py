"""Utilities to flatten neighbor lists into CSR-style arrays for numba kernels."""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .neighborhood import BoundarySample


def flatten_neighbors(neighbors: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten list-of-lists neighbor indices into (flat, offsets).

    Returns:
        flat: int32 array of all neighbor indices concatenated.
        offsets: int32 array of length (n+1) where offsets[i]:offsets[i+1] is neighbors of i.
    """
    counts = [len(lst) for lst in neighbors]
    n = len(neighbors)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    total = offsets[-1]
    flat = np.empty(total, dtype=np.int32)
    pos = 0
    for lst in neighbors:
        length = len(lst)
        if length:
            flat[pos : pos + length] = lst
        pos += length
    return flat, offsets


def flatten_boundary_neighbors(boundary: List[List[BoundarySample]]):
    """Flatten boundary neighbor samples per particle into CSR arrays.

    Returns:
        b_pos_flat: float32 array (M, 3)
        b_mass_flat: float32 array (M,)
        b_offsets: int32 array (n+1)
    """
    counts = [len(lst) for lst in boundary]
    n = len(boundary)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    total = offsets[-1]
    if total == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            offsets,
        )
    pos_flat = np.empty((total, 3), dtype=np.float32)
    mass_flat = np.empty((total,), dtype=np.float32)
    idx = 0
    for lst in boundary:
        for sample in lst:
            pos_flat[idx, 0] = sample.position[0]
            pos_flat[idx, 1] = sample.position[1]
            pos_flat[idx, 2] = sample.position[2]
            mass_flat[idx] = sample.pseudo_mass
            idx += 1
    return pos_flat, mass_flat, offsets
