#!/usr/bin/env python
"""Extract frames from simulation output by stride or target FPS.

This script copies selected frame directories from a source folder to a destination
folder with sequential numbering, suitable for video encoding or Blender rendering.

Usage:
    python scripts/extract_frames.py --input outputs/frames --output outputs/show_frames --stride 10
    python scripts/extract_frames.py --input outputs/frames --output outputs/show_frames --target-fps 60 --sim-fps 100
    python scripts/extract_frames.py --input outputs/frames --output outputs/show_frames --target-fps 60 --time-step 0.01
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def get_sorted_frame_dirs(input_dir: Path) -> list[tuple[int, Path]]:
    """Get sorted list of (step_number, directory_path) from input directory."""
    frames = []
    for d in input_dir.iterdir():
        if d.is_dir():
            try:
                step = int(d.name)
                frames.append((step, d))
            except ValueError:
                continue
    frames.sort(key=lambda x: x[0])
    return frames


def extract_frames(
    input_dir: Path,
    output_dir: Path,
    stride: int = 1,
    target_fps: float | None = None,
    sim_fps: float | None = None,
    time_step: float | None = None,
) -> int:
    """Extract frames with given stride or target FPS.
    
    Args:
        input_dir: Source directory containing numbered frame folders
        output_dir: Destination directory for extracted frames
        stride: Take every Nth frame (default: 1, no skipping)
        target_fps: Desired output FPS (requires sim_fps or time_step)
        sim_fps: Simulation FPS (1 / time_step)
        time_step: Simulation time step in seconds
    
    Returns:
        Number of frames extracted
    """
    # Calculate stride from target FPS if provided
    if target_fps is not None:
        if sim_fps is None and time_step is None:
            raise ValueError("target-fps requires either --sim-fps or --time-step")
        if sim_fps is None:
            sim_fps = 1.0 / time_step
        calculated_stride = max(1, round(sim_fps / target_fps))
        stride = max(stride, calculated_stride)
        print(f"Simulation FPS: {sim_fps:.1f}, Target FPS: {target_fps:.1f}, Stride: {stride}")
    
    # Get all frame directories
    frames = get_sorted_frame_dirs(input_dir)
    if not frames:
        raise FileNotFoundError(f"No frame directories found in {input_dir}")
    
    print(f"Found {len(frames)} frames in {input_dir}")
    
    # Apply stride
    selected_frames = frames[::stride]
    print(f"Extracting {len(selected_frames)} frames (stride={stride})")
    
    # Clean and recreate output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected frames with sequential numbering
    for output_idx, (step, src_dir) in enumerate(tqdm(selected_frames, desc="Copying frames")):
        dst_dir = output_dir / f"{output_idx:05d}"
        shutil.copytree(src_dir, dst_dir)
    
    print(f"Output: {output_dir}")
    print(f"Extracted {len(selected_frames)} frames")
    return len(selected_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from simulation output by stride or target FPS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract every 10th frame
  python scripts/extract_frames.py -i outputs/frames -o outputs/show_frames --stride 10

  # Extract for 60 FPS output (simulation at 100 FPS)
  python scripts/extract_frames.py -i outputs/frames -o outputs/show_frames --target-fps 60 --sim-fps 100

  # Extract for 60 FPS output (time_step = 0.01s means 100 FPS)
  python scripts/extract_frames.py -i outputs/frames -o outputs/show_frames --target-fps 60 --time-step 0.01
        """
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Input directory containing numbered frame folders")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth frame (default: 1)")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Target output FPS (requires --sim-fps or --time-step)")
    parser.add_argument("--sim-fps", type=float, default=None,
                        help="Simulation FPS")
    parser.add_argument("--time-step", type=float, default=None,
                        help="Simulation time step in seconds (sim_fps = 1/time_step)")
    
    args = parser.parse_args()
    
    extract_frames(
        input_dir=args.input,
        output_dir=args.output,
        stride=args.stride,
        target_fps=args.target_fps,
        sim_fps=args.sim_fps,
        time_step=args.time_step,
    )


if __name__ == "__main__":
    main()
