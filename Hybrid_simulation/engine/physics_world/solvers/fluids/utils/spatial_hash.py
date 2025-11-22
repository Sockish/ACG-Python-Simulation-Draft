"""Uniform spatial hash grid for neighbor queries."""

from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import DefaultDict, Dict, Iterable, List, Tuple

from ....math_utils import Vec3

Cell = Tuple[int, int, int]


class SpatialHashGrid:
    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size if cell_size > 0 else 0.0
        self.cells: DefaultDict[Cell, List[int]] = defaultdict(list)

    def build(self, positions: Iterable[Vec3]) -> None:
        self.cells.clear()
        for idx, pos in enumerate(positions):
            self.cells[self._cell(pos)].append(idx)

    def neighbors(self, position: Vec3) -> Iterable[int]:
        base = self._cell(position)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = (base[0] + dx, base[1] + dy, base[2] + dz)
                    yield from self.cells.get(cell, [])

    def _cell(self, position: Vec3) -> Cell:
        return (
            floor(position[0] * self.inv_cell),
            floor(position[1] * self.inv_cell),
            floor(position[2] * self.inv_cell),
        )
