"""Particle containers and neighbor search utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ParticleSystem:
    """Stores all per-particle properties as NumPy arrays for vector ops."""

    count: int
    mass: float

    def __post_init__(self) -> None:
        self.positions = np.zeros((self.count, 3), dtype=np.float32)
        self.velocities = np.zeros_like(self.positions)
        self.accelerations = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.densities = np.zeros(self.count, dtype=np.float32)
        self.pressures = np.zeros(self.count, dtype=np.float32)

    def seed_block(
        self,
        block_min: Tuple[float, float, float],
        block_max: Tuple[float, float, float],
        jitter: float = 0.002,
    ) -> None:
        lin = np.linspace(0.0, 1.0, int(round(self.count ** (1 / 3))) + 2)
        grid = np.array(np.meshgrid(lin, lin, lin)).T.reshape(-1, 3)
        rng = np.random.default_rng(42)
        span = np.array(block_max) - np.array(block_min)
        pts = np.array(block_min) + grid * span
        pts = pts[: self.count]
        pts += rng.uniform(-jitter, jitter, size=pts.shape)
        self.positions[: len(pts)] = pts

    def reset_forces(self) -> None:
        self.forces.fill(0.0)


class SpatialHash:
    """Uniform grid for approximate O(n) neighbor lookup."""

    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}

    def _cell_coords(self, position: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(position * self.inv_cell).astype(int))

    def build(self, positions: np.ndarray) -> None:
        self.cells.clear()
        for idx, pos in enumerate(positions):
            key = self._cell_coords(pos)
            self.cells.setdefault(key, []).append(idx)

    def iter_neighbors(self, position: np.ndarray) -> Iterable[int]:
        base = self._cell_coords(position)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (base[0] + dx, base[1] + dy, base[2] + dz)
                    if key in self.cells:
                        yield from self.cells[key]

