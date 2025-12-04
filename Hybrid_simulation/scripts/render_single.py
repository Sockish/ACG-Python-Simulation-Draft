import os
import shutil
import subprocess
import argparse
from tqdm import tqdm

## 单线程渲染脚本
## 用法: python scripts/render_single.py --scene_file test.blend --input_dir config/outputs/show_frames
## 渲染结果将保存到:
##   - 每帧文件夹: config/outputs/show_frames/00000/render.png
##   - 汇总文件夹: config/outputs/renders/00000.png (便于导出视频)

BLENDER_PATH = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RENDERING_SCRIPT = os.path.join(SCRIPT_DIR, "rendering_script.py")


def process_frame(frame_dir, frame_name, args, renders_dir):
    """处理单帧渲染"""
    # 帧文件夹内的输出路径
    frame_output = os.path.join(frame_dir, args.rendered_image_name)
    # renders文件夹内的输出路径（以帧号命名）
    renders_output = os.path.join(renders_dir, f"{frame_name}.png")
    
    cmd = [
        BLENDER_PATH,
        '-b', args.scene_file,
        '--python', RENDERING_SCRIPT,
        '--',
        args.device_type,
        '0',  # 单GPU，使用索引0
        frame_dir,
        frame_output
    ]
    
    if args.quiet:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd)
    
    # 复制渲染图片到renders文件夹
    if os.path.exists(frame_output):
        shutil.copy2(frame_output, renders_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='单线程Blender渲染脚本')
    parser.add_argument('--scene_file', type=str, required=True, help='Blender场景文件路径')
    parser.add_argument('--rendered_image_name', type=str, default='render.png', help='渲染图片文件名')
    parser.add_argument('--input_dir', type=str, required=True, help='帧文件夹目录')
    parser.add_argument('--device_type', type=str, default='OPTIX', help='渲染设备类型 (OPTIX/CUDA/HIP)')
    parser.add_argument('--quiet', action='store_true', help='静默模式，不显示Blender输出')
    parser.add_argument('--filter_by', type=str, default=None,
                        help='只渲染包含此文件的帧 (例如 "liquid_surface.obj")')
    parser.add_argument('--renders_dir', type=str, default=None,
                        help='渲染结果汇总目录 (默认为 input_dir/../renders)')

    args = parser.parse_args()
    
    # 转换为绝对路径
    args.input_dir = os.path.abspath(args.input_dir)
    args.scene_file = os.path.abspath(args.scene_file)
    
    # 设置renders目录
    if args.renders_dir:
        renders_dir = os.path.abspath(args.renders_dir)
    else:
        renders_dir = os.path.join(os.path.dirname(args.input_dir), "renders")
    
    # 清理并创建renders目录
    if os.path.exists(renders_dir):
        shutil.rmtree(renders_dir)
    os.makedirs(renders_dir, exist_ok=True)
    print(f"渲染结果将汇总到: {renders_dir}")
    
    # 获取帧列表
    frame_list = os.listdir(args.input_dir)
    
    # 过滤帧（如果指定了--filter_by）
    if args.filter_by:
        frame_list = [
            f for f in frame_list 
            if os.path.isdir(os.path.join(args.input_dir, f)) and
               os.path.exists(os.path.join(args.input_dir, f, args.filter_by))
        ]
        print(f"过滤后共 {len(frame_list)} 帧包含 '{args.filter_by}'")
    
    frame_list.sort(key=lambda x: int(x))
    num_frames = len(frame_list)

    print(f"共 {num_frames} 帧待渲染")
    print(f"渲染设备: {args.device_type}")
    print("-" * 50)

    # 单线程顺序渲染
    for frame in tqdm(frame_list, desc="渲染进度"):
        frame_dir = os.path.join(args.input_dir, frame)
        process_frame(frame_dir, frame, args, renders_dir)

    print(f"\n渲染完成! 所有图片已汇总到: {renders_dir}")
