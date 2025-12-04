
## Simulation and Rendering Instructions
.venv/Scripts/python.exe ./scripts/simulate.py --steps 800

python scripts/reconstruct.py --config config/scene_config.yaml --target-fps 100

.venv/Scripts/python.exe ./scripts/extract_frames.py -i config/outputs/frames -o config/outputs/show_frames --target-fps 10 --time-step 0.01
.
python scripts/render.py --scene_file test.blend --input_dir config/outputs/show_frames

or using: 

python scripts/render_single.py --scene_file test.blend --input_dir config/outputs/show_frames

## Combine rendered frames into a video using ffmpeg
ffmpeg -framerate 60 -i config/outputs/renders/%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4