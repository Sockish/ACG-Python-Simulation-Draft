import os
import shutil
import subprocess
import re
import multiprocessing as mp
from tqdm import tqdm
import argparse

## usage: python scripts/render.py --scene_file test.blend --input_dir config/outputs/show_frames
## Renders will be saved to:
##   - Each frame folder: config/outputs/show_frames/00000/render.png
##   - Collected folder:  config/outputs/renders/00000.png (for easy video export)

def get_visible_gpu_indices():
    # Read the CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if cuda_visible_devices is None:
        # If the environment variable is not set, all GPUs are visible
        return None
    elif cuda_visible_devices.strip() == "":
        # If the environment variable is set to an empty string, no GPUs are visible
        return []
    else:
        # Split the environment variable by comma and convert to integers
        return [int(gpu.strip()) for gpu in cuda_visible_devices.split(',')]

def get_gpu_count():
    try:
        # Get visible GPU indices from the environment variable
        visible_gpu_indices = get_visible_gpu_indices()

        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Decode the output
        output = result.stdout.decode('utf-8')

        # Count the number of lines in the output
        total_gpus = len(re.findall(r'.+\n', output))

        if visible_gpu_indices is None:
            # If no environment variable is set, all GPUs are visible
            return total_gpus
        else:
            # Filter the GPU indices based on the environment variable
            return len([i for i in visible_gpu_indices if i < total_gpus])
    except Exception as e:
        print("An error occurred: ", e)
        return 0



# define a template bash command that will be run by process.
# this command will be run in the shell
BLENDER_PATH = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RENDERING_SCRIPT = os.path.join(SCRIPT_DIR, "rendering_script.py")
# Use list-based command execution to avoid shell escaping issues
num_gpus = get_gpu_count()
print("Number of Visible GPUs:", num_gpus)

def process_frame(frame_dir, frame_name, rank, args, renders_dir):
    # Output path in frame folder
    frame_output = os.path.join(frame_dir, args.rendered_image_name)
    # Output path in renders folder (named by frame number)
    renders_output = os.path.join(renders_dir, f"{frame_name}.png")
    
    cmd = [
        BLENDER_PATH,
        '-b', args.scene_file,
        '--python', RENDERING_SCRIPT,
        '--',
        args.device_type,
        str(rank % num_gpus),
        frame_dir,
        frame_output
    ]
    if rank == 0 and not args.quiet:
        subprocess.run(cmd)
    else:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Copy rendered image to renders folder
    if os.path.exists(frame_output):
        shutil.copy2(frame_output, renders_output)

def worker(frame_dir, frame_name, rank, args, renders_dir):
    try:
        process_frame(frame_dir, frame_name, rank, args, renders_dir)
    except Exception as e:
        print(f"failed to process {frame_dir}")
        print(e)
    return 1 # return 1 to indicate success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, required=True)
    parser.add_argument('--rendered_image_name', type=str, default='render.png')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device_type', type=str, default='OPTIX')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--filter_by', type=str, default=None,
                        help='Only render frames containing this file (e.g., "liquid_surface.obj"). '
                             'Useful when reconstruct.py was run with --target-fps.')
    parser.add_argument('--renders_dir', type=str, default=None,
                        help='Directory to collect all rendered PNGs (defaults to input_dir/../renders)')

    args = parser.parse_args()
    
    # Convert input_dir and scene_file to absolute paths
    args.input_dir = os.path.abspath(args.input_dir)
    args.scene_file = os.path.abspath(args.scene_file)
    
    # Set up renders directory (for collecting all PNGs)
    if args.renders_dir:
        renders_dir = os.path.abspath(args.renders_dir)
    else:
        # Default: sibling folder to input_dir called "renders"
        renders_dir = os.path.join(os.path.dirname(args.input_dir), "renders")
    
    # Clean and create renders directory
    if os.path.exists(renders_dir):
        shutil.rmtree(renders_dir)
    os.makedirs(renders_dir, exist_ok=True)
    print(f"Renders will be collected in: {renders_dir}")
    
    frame_list = os.listdir(args.input_dir)
    
    # Filter frames if --filter_by is specified
    if args.filter_by:
        frame_list = [
            f for f in frame_list 
            if os.path.isdir(os.path.join(args.input_dir, f)) and
               os.path.exists(os.path.join(args.input_dir, f, args.filter_by))
        ]
        print(f"Filtered to {len(frame_list)} frames containing '{args.filter_by}'")
    
    frame_list.sort(key=lambda x: int(x))
    num_frames = len(frame_list)

    print(f"Processing {num_frames} frames with {args.num_workers} workers")
    print(f"Using device type: {args.device_type}")

    


    # Using a pool of workers to process the images
    pool = mp.Pool(args.num_workers)

    # Progress bar setup
    pbar = tqdm(total=len(frame_list))

    # Update progress bar in callback
    def update_pbar(result):
        pbar.update(1)


    for i, frame in enumerate(frame_list):
        frame_dir = os.path.join(args.input_dir, frame)
        rank = i % args.num_workers
        pool.apply_async(worker, args=(frame_dir, frame, rank, args, renders_dir), callback=update_pbar)

    
    pool.close()
    pool.join()
    pbar.close()

    print(f"\nRendering complete! All PNGs collected in: {renders_dir}")
