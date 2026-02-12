# 切换到独立环境
# python3 -m venv venv
# source venv/bin/activate
# pip3 install -r requirements.txt
import cv2
import torch
import numpy as np
import os
import argparse
from depth_anything_v2.dpt import DepthAnythingV2

# === 配置 ===
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
# 这里的 encoder 必须和你下载的模型文件名一致 (vits, vitb, vitl)
MODEL_CONFIG = {
    'encoder': 'vits', 
    'features': 64, 
    'out_channels': [48, 96, 192, 384]
}
MODEL_PATH = 'checkpoints/depth_anything_v2_vits.pth' # 确保路径正确

def extract_and_process(video_path, frame_index, output_dir):
    # 1. 初始化模型
    print(f"🚀 Loading model to {DEVICE}...")
    depth_model = DepthAnythingV2(**MODEL_CONFIG)
    depth_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()

    # 2. 读取视频指定帧
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_index >= total_frames:
        print(f"❌ Error: Frame {frame_index} out of bounds (Total: {total_frames})")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Error: Could not read frame.")
        return

    h, w = frame.shape[:2]
    print(f"📸 Captured Frame {frame_index} | Resolution: {w}x{h}")

    # 3. 处理接缝 (The Padding Trick)
    # 左右各扩充 10% 的内容，让模型知道边界是连续的
    pad_w = int(w * 0.1) 
    # numpy切片: [所有行, 左侧pad_w列]
    left_part = frame[:, 0:pad_w] 
    # numpy切片: [所有行, 右侧最后pad_w列]
    right_part = frame[:, w-pad_w:w] 

    # 拼接: [右边末尾] + [原始图片] + [左边开头]
    padded_frame = np.concatenate((right_part, frame, left_part), axis=1)

    # 4. 深度推理
    # input_size=1024 或 2048 能获得更精细的纹理，但速度变慢
    # Mac M4 建议尝试 1024 或 1536
    depth = depth_model.infer_image(padded_frame, input_size=1024)

    # 5. 后处理与裁切
    # 归一化到 0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_uint8 = depth_normalized.astype(np.uint8)

    # 把之前扩充的 10% 切掉，只保留中间原本的部分
    # 注意：infer_image 输出的尺寸通常和输入一致，但为了保险，我们按比例裁切
    out_h, out_w = depth_uint8.shape
    real_w = out_w - (pad_w * 2)
    # 这里的裁切要非常小心，确保像素对齐
    # 由于 infer_image 可能会有 resize 行为，最好是 resize 回 padded 尺寸再 crop
    # 但 DepthAnythingV2 的 infer_image 返回的是原图大小的 numpy 数组，所以直接 crop 即可
    
    final_depth = depth_uint8[:, pad_w : out_w - pad_w]
    
    # 确保尺寸严格匹配原图 (应对可能的舍入误差)
    final_depth = cv2.resize(final_depth, (w, h))

    # 6. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 保存原图 (作为纹理)
    cv2.imwrite(os.path.join(output_dir, f"{name}_f{frame_index}_RGB.jpg"), frame)
    
    # 保存深度图 (作为 Displacement/Depth)
    cv2.imwrite(os.path.join(output_dir, f"{name}_f{frame_index}_Depth.png"), final_depth)
    
    # 保存合成预览 (上下排列)
    depth_color = cv2.cvtColor(final_depth, cv2.COLOR_GRAY2BGR)
    preview = np.vstack((frame, depth_color))
    cv2.imwrite(os.path.join(output_dir, f"{name}_f{frame_index}_Preview.jpg"), preview)

    print(f"✅ Done! Files saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to insv/mp4 video')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to extract')
    parser.add_argument('--out', type=str, default='./out', help='Output folder')
    
    args = parser.parse_args()
    extract_and_process(args.video, args.frame, args.out)