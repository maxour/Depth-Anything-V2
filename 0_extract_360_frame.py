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
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

def extract_and_process(video_path, frame_index, output_dir, encoder):
    # 1. 初始化模型
    print(f"🚀 Loading model to {DEVICE}...")
    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
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
    # 左右各扩充 10% 的内容
    pad_w = int(w * 0.1) 
    left_part = frame[:, 0:pad_w] 
    right_part = frame[:, w-pad_w:w] 
    padded_frame = np.concatenate((right_part, frame, left_part), axis=1)

    # === 关键修改开始 ===
    
    # 获取 Padding 后的原始尺寸
    orig_pad_h, orig_pad_w = padded_frame.shape[:2]

    # [步骤 A] 手动缩放 (Force Resize)
    # 强制将图片缩小到模型能处理的大小 (例如宽 1024 或 1176 - 最好是14的倍数)
    # 对于 Mac Air，推荐使用 1024 以保证不爆显存
    infer_width = 1024 
    ratio = infer_width / orig_pad_w
    infer_height = int(orig_pad_h * ratio)
    
    # 确保高度是 14 的倍数 (ViT 模型对 Patch 对齐有要求，虽然库通常会处理，但手动做更稳)
    infer_height = (infer_height // 14) * 14
    infer_width = (infer_width // 14) * 14
    
    resized_padded_frame = cv2.resize(padded_frame, (infer_width, infer_height))

    print(f"📉 Resizing for inference: {orig_pad_w}x{orig_pad_h} -> {infer_width}x{infer_height}")

    # 4. 深度推理
    # 注意：这里我们传入已经缩小的图片，input_size 参数可以省略或保持一致
    with torch.no_grad():
        depth = depth_model.infer_image(resized_padded_frame, input_size=infer_width)

    # 5. 后处理与裁切
    # [步骤 B] 恢复尺寸 (Upscale back)
    # 将生成的低分辨率深度图放大回 padded 的原始尺寸
    depth = cv2.resize(depth, (orig_pad_w, orig_pad_h), interpolation=cv2.INTER_CUBIC)

    # 归一化到 0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_uint8 = depth_normalized.astype(np.uint8)

    # 把之前扩充的 10% 切掉，只保留中间原本的部分
    # 此时 depth_uint8 的尺寸已经回到了 orig_pad_w x orig_pad_h
    final_depth = depth_uint8[:, pad_w : orig_pad_w - pad_w]
    
    # [步骤 C] 双重保险：确保最终尺寸严格匹配原视频帧
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
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    extract_and_process(args.video, args.frame, args.out, args.encoder)