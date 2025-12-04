"""Scene configuration dataclasses and loader utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import List, Sequence


_YAML_MODULE: ModuleType | None = None


def _load_yaml_module() -> ModuleType:
    global _YAML_MODULE
    if _YAML_MODULE is None:
        try:
            _YAML_MODULE = import_module("yaml")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "PyYAML is required to load scene configurations. Install it via 'pip install pyyaml'."
            ) from exc
    return _YAML_MODULE


@dataclass
class LiquidBoxConfig:
    min_corner: Sequence[float]  # meters (m)
    max_corner: Sequence[float]  # meters (m)
    particle_spacing: float  # meters (m)
    rest_density: float  # kilograms per cubic meter (kg/m^3)
    smoothing_length: float  # meters (m)


@dataclass
class RigidBodyConfig:
    name: str
    mesh_path: Path
    mass: float  # kilograms (kg)
    inertia: Sequence[float]  # kilogram meter squared (kg·m^2)
    initial_position: Sequence[float]  # meters (m)
    initial_orientation: Sequence[float]  # unit quaternion (dimensionless)
    initial_linear_velocity: Sequence[float]  # meters per second (m/s)
    initial_angular_velocity: Sequence[float]  # radians per second (rad/s)


@dataclass
class StaticBodyConfig:
    name: str
    mesh_path: Path
    initial_position: Sequence[float]  # meters (m)
    initial_orientation: Sequence[float]  # unit quaternion


@dataclass
class SimulationConfig:
    time_step: float  # seconds (s)
    total_steps: int
    gravity: Sequence[float]  # meters per second squared (m/s^2)


@dataclass
class ExportConfig:
    output_root: Path
    fluid_subdir: str = "fluid"
    rigid_subdir: str = "rigid"
    static_subdir: str = "static"

    def fluid_dir(self) -> Path:
        return self.output_root / self.fluid_subdir

    def rigid_dir(self) -> Path:
        return self.output_root / self.rigid_subdir

    def static_dir(self) -> Path:
        return self.output_root / self.static_subdir


@dataclass
class SceneConfig:
    scene_name: str
    simulation: SimulationConfig
    liquid_box: LiquidBoxConfig
    rigid_bodies: List[RigidBodyConfig] = field(default_factory=list)
    static_bodies: List[StaticBodyConfig] = field(default_factory=list)
    export: ExportConfig | None = None


def _coerce_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path)


def load_scene_config(config_path: str | Path) -> SceneConfig:
    """Load a scene configuration from YAML."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        yaml_module = _load_yaml_module()
        raw = yaml_module.safe_load(handle)

    base_dir = path.parent

    simulation = SimulationConfig(**raw["simulation"])
    if raw.get("liquid_box"):
        liquid_box = LiquidBoxConfig(**raw["liquid_box"])
    else:
        print("Warning: No liquid box configuration found, will use no fluid simulation.")
        liquid_box = None

    rigid_bodies = [
        RigidBodyConfig(
            name=entry["name"],
            mesh_path=_coerce_path(base_dir, entry["mesh_path"]),
            mass=entry["mass"],
            inertia=entry["inertia"],
            initial_position=entry["initial_position"],
            initial_orientation=entry["initial_orientation"],
            initial_linear_velocity=entry.get("initial_linear_velocity", (0.0, 0.0, 0.0)),
            initial_angular_velocity=entry.get("initial_angular_velocity", (0.0, 0.0, 0.0)),
        )
        for entry in (raw.get("rigid_bodies") or [])
    ]

    static_bodies = [
        StaticBodyConfig(
            name=entry["name"],
            mesh_path=_coerce_path(base_dir, entry["mesh_path"]),
            initial_position=entry.get("initial_position", (0.0, 0.0, 0.0)),
            initial_orientation=entry.get("initial_orientation", (0.0, 0.0, 0.0, 1.0)),
        )
        for entry in (raw.get("static_bodies") or [])
    ]

    export_cfg = raw.get("export")
    export = None
    if export_cfg:
        export = ExportConfig(
            output_root=_coerce_path(base_dir, export_cfg["output_root"]),
            fluid_subdir=export_cfg.get("fluid_subdir", "fluid"),
            rigid_subdir=export_cfg.get("rigid_subdir", "rigid"),
            static_subdir=export_cfg.get("static_subdir", "static"),
        )

    return SceneConfig(
        scene_name=raw.get("scene_name", path.stem),
        simulation=simulation,
        liquid_box=liquid_box,
        rigid_bodies=rigid_bodies,
        static_bodies=static_bodies,
        export=export,
    )
