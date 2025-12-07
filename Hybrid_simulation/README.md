
## Simulation and Rendering Instructions
.venv/Scripts/python.exe ./scripts/simulate.py --steps 800

.venv/Scripts/python.exe scripts/simulate.py --use-taichi --steps 10

python scripts/reconstruct.py --config config/scene_config.yaml --target-fps 60

.venv/Scripts/python.exe ./scripts/extract_frames.py -i config/outputs_nailong_matrix/frames -o config/outputs_nailong_matrix/show_frames --target-fps 50 --time-step 0.01
.
Better Don't use !!!!!!!!python scripts/render.py --scene_file dambreak.blend --input_dir config/output_dambreak/show_frames

python scripts/render_single.py --scene_file liquid.blend --input_dir config/outputs_pure_liquid2/show_frames

## Combine rendered frames into a video using ffmpeg
ffmpeg -framerate 60 -i config/outputs_pure_liquid2/renders/%05d.png -c:v libx264 -pix_fmt yuv420p outputs_pure_liquid2.mp4