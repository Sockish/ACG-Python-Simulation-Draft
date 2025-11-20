"""Utilities for handing data off to Blender for rendering."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def launch_blender(
    blender_executable: str,
    blender_script: Path,
    manifest_path: Path,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Invoke Blender in background mode with the given script."""

    command = [
        blender_executable,
        "-b",
        "--python",
        str(blender_script),
        "--",
        str(manifest_path),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)

