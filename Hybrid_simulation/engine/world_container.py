"""High-level orchestration layer around the physics world and exporters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .configuration import SceneConfig, load_scene_config
from .exporter import SimulationExporter
from .physics_world.world import PhysicsWorld


@dataclass
class WorldContainer:
    """Bundles scene configuration, physics world, and exporter."""

    config: SceneConfig
    world: PhysicsWorld
    exporter: SimulationExporter
    current_step: int = 0

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "WorldContainer":
        config = load_scene_config(config_path)
        world = PhysicsWorld.from_config(config)
        exporter = SimulationExporter.from_config(config.export)
        return cls(config=config, world=world, exporter=exporter)

    def step(self, dt: float | None = None, *, export: bool = True) -> None:
        """Advance the world by a single step and optionally export state."""
        dt = dt if dt is not None else self.config.simulation.time_step
        if self.current_step < 100:
            liquid_force_damp = 0.1
        else:
            liquid_force_damp = 1.0
        snapshot = self.world.step(liquid_force_damp, dt)
        if export:
            self.exporter.export_step(self.current_step, snapshot)
        self.current_step += 1

    def run(self, steps: Optional[int] = None) -> None:
        """Execute multiple simulation steps."""
        total_steps = steps if steps is not None else self.config.simulation.total_steps
        for _ in range(total_steps):
            self.step()
