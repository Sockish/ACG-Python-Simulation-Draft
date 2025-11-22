"""Fluid particle containers and neighbor search utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ParticleSystem:
    """Stores all per-particle properties as NumPy arrays for vector ops."""

    count: int
    _seed_cursor: int = field(init=False, default=0)

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
        max_particles: int | None = None,
    ) -> int:
        """Fill particle positions within a block, optionally capping the count."""

        remaining = self.count - self._seed_cursor
        if remaining <= 0:
            return 0
        target = remaining if max_particles is None else min(remaining, max_particles)
        if target <= 0:
            return 0
        axis_samples = int(np.ceil(target ** (1 / 3))) + 2
        lin = np.linspace(0.0, 1.0, axis_samples)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
        span = np.array(block_max) - np.array(block_min)
        pts = np.array(block_min) + grid * span
        pts = pts[:target]
        rng = np.random.default_rng()
        if pts.size:
            pts += rng.uniform(-jitter, jitter, size=pts.shape)
        start = self._seed_cursor
        end = start + len(pts)
        self.positions[start:end] = pts
        self._seed_cursor = end
        return len(pts)

    def reset_forces(self) -> None:
        self.forces.fill(0.0)

    @property
    def seeded_count(self) -> int:
        return self._seed_cursor


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
