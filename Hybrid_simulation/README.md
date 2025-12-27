
## Simulation and Rendering Instructions
.venv/Scripts/python.exe ./scripts/simulate.py --steps 800

.venv/Scripts/python.exe scripts/simulate.py --use-taichi --steps 10

python scripts/reconstruct.py --config config/scene_config.yaml --target-fps 60

.venv/Scripts/python.exe ./scripts/extract_frames.py -i config/outputs_nailong_matrix/frames -o config/outputs_nailong_matrix/show_frames --target-fps 50 --time-step 0.01
.
Better Don't use !!!!!!!!python scripts/render.py --scene_file dambreak.blend --input_dir config/output_dambreak/show_frames

python scripts/render_single.py --scene_file dambreak.blend --input_dir config/outputs_simulate_mpm/show_frames

## Combine rendered frames into a video using ffmpeg
ffmpeg -framerate 60 -i config/outputs_simulate_mpm/renders/%05d.png -c:v libx264 -pix_fmt yuv420p outputs_simulate_mpm.mp4


## MPM Simulation Instructions
python ./mpm_instances/simulate_mpm4.py

Then just render the output frames using the render_single.py script above.

python scripts/render_single.py --scene_file Landscape.blend --input_dir config/outputs_simulate_mpm5/show_frames

python scripts/render_single.py --scene_file floor.blend --input_dir config/outputs_simulate_mpm4/show_frames

python scripts/render_single.py --scene_file Landscape.blend --input_dir config/outputs_simulate_mpm3/show_frames

python scripts/render_single.py --scene_file dambreak.blend --input_dir config/outputs_simulate_mpm3/show_frames