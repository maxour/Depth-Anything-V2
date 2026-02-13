# 切换到独立环境
# python3 -m venv venv
# source venv/bin/activate
# pip3 install -r requirements.txt
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def generate_depth_map(image_path, output_path, model_type='vitl'):
    """
    针对 ViviMe 流程 STEP 0/7 的深度图生成函数
    :param image_path: 输入 JPG/PNG 路径
    :param output_path: 输出深度图路径
    :param model_type: 模型大小 ('vits', 'vitb', 'vitl' - 推荐 vitl 以获得最高精度)
    """
    
    # 1. 硬件配置：优先使用 Mac M4 的 GPU (MPS)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 2. 模型初始化 (根据不同版本调整参数)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    model = DepthAnythingV2(**model_configs[model_type])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{model_type}.pth', map_location='cpu'))
    model.to(device).eval()

    # 3. 读取并预处理图片
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error: 无法读取图片 {image_path}")
        return

    # 4. 执行推理
    with torch.no_grad():
        # DepthAnything V2 直接返回预测的深度图
        depth = model.infer_image(raw_img) # 结果为 HxW 的 numpy 数组

    # 5. 后处理：归一化到 0-255 以便保存为灰度图
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    # 6. 保存结果
    # 可选：应用伪彩色（Color Map）以便于人类视觉观察深度层次
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(output_path, depth_norm) # 保存为标准灰度深度图
    cv2.imwrite(output_path.replace('.png', '_color.png'), depth_color) # 保存为彩色深度图
    
    print(f"✅ 深度图已生成至: {output_path}")

# 使用示例
if __name__ == "__main__":
    generate_depth_map('input.jpg', 'output_depth.png')