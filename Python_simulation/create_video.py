"""Convert PNG sequence to MP4 video."""
import argparse
from pathlib import Path
import subprocess
import sys


def create_video_with_ffmpeg(
    png_dir: Path,
    output_path: Path,
    fps: int = 30,
    quality: str = "high"
) -> None:
    """Create MP4 video from PNG sequence using FFmpeg.
    
    Args:
        png_dir: Directory containing frame_*.png files
        output_path: Output video file path
        fps: Frames per second (default: 30)
        quality: 'high', 'medium', or 'low' (affects CRF value)
    """
    png_files = sorted(png_dir.glob("frame_*.png"))
    
    if not png_files:
        raise FileNotFoundError(f"No frame_*.png files found in {png_dir}")
    
    print(f"Found {len(png_files)} frames in {png_dir}")
    print(f"Creating video at {fps} FPS with {quality} quality...")
    
    # Quality presets (CRF: lower = better quality, higher file size)
    quality_map = {
        "high": 18,
        "medium": 23,
        "low": 28
    }
    crf = quality_map.get(quality.lower(), 23)
    
    # FFmpeg command
    # -framerate: input framerate
    # -i: input pattern
    # -c:v libx264: H.264 codec
    # -crf: quality (0-51, lower is better)
    # -pix_fmt yuv420p: compatibility format
    # -y: overwrite output
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(png_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-y",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Video created successfully: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg error: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please install FFmpeg:", file=sys.stderr)
        print("  Windows: choco install ffmpeg  (or download from ffmpeg.org)", file=sys.stderr)
        print("  Linux: sudo apt-get install ffmpeg", file=sys.stderr)
        print("  macOS: brew install ffmpeg", file=sys.stderr)
        raise


def create_video_with_opencv(
    png_dir: Path,
    output_path: Path,
    fps: int = 30
) -> None:
    """Create MP4 video from PNG sequence using OpenCV (fallback method).
    
    Args:
        png_dir: Directory containing frame_*.png files
        output_path: Output video file path
        fps: Frames per second (default: 30)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
    
    png_files = sorted(png_dir.glob("frame_*.png"))
    
    if not png_files:
        raise FileNotFoundError(f"No frame_*.png files found in {png_dir}")
    
    print(f"Found {len(png_files)} frames in {png_dir}")
    print(f"Creating video at {fps} FPS using OpenCV...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(png_files[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for png_file in png_files:
        frame = cv2.imread(str(png_file))
        video.write(frame)
    
    video.release()
    print(f"✓ Video created successfully: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG sequence to MP4 video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_video.py
  python create_video.py --fps 60 --quality high
  python create_video.py --input output/render --output my_video.mp4
  python create_video.py --method opencv
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("C:/output/render"),
        help="Directory containing frame_*.png files (default: output/render)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/simulation.mp4"),
        help="Output video file path (default: output/simulation.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "low"],
        default="medium",
        help="Video quality (default: medium, only for FFmpeg)"
    )
    parser.add_argument(
        "--method",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="Encoding method (default: ffmpeg, fallback: opencv)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.method == "ffmpeg":
            create_video_with_ffmpeg(args.input, args.output, args.fps, args.quality)
        else:
            create_video_with_opencv(args.input, args.output, args.fps)
    except Exception as e:
        print(f"Error creating video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
