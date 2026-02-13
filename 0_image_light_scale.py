# 切换到独立环境
# python3 -m venv venv
# source venv/bin/activate
import cv2
import numpy as np
import json
import argparse
import os

def analyze_scene(rgb_path, depth_path, output_dir):
    """
    综合分析场景光照和深度信息
    """
    # 1. 读取图像
    rgb_img = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) # 读取单通道灰度
    
    if rgb_img is None or depth_img is None:
        print("❌ Error: 无法读取 RGB 或 Depth 图片")
        return

    h, w = rgb_img.shape[:2]
    print(f"🖼  Processing: {w}x{h}")

    # ==========================
    # Part 1: 光照分析 (Lighting)
    # ==========================
    
    # A. 寻找主光源 (Sun Position)
    # 转为灰度图
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊：去除噪点，让光源中心更聚拢 (核大小 41x41)
    blurred = cv2.GaussianBlur(gray, (41, 41), 0)
    # 寻找最大值位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    
    sun_x, sun_y = max_loc
    
    # B. 计算环境光 (Ambient Color)
    # 计算全图平均颜色 (BGR -> RGB)
    avg_color_bgr = np.mean(rgb_img, axis=(0, 1))
    ambient_rgb = [int(avg_color_bgr[2]), int(avg_color_bgr[1]), int(avg_color_bgr[0])]

    # ==========================
    # Part 2: 深度/缩放计算工具
    # ==========================
    
    # 定义一个内部函数，用于模拟“点击查询”
    def get_scale_at(u, v, base_scale=1.0):
        """
        输入 UV 坐标 (0-1)，返回推荐缩放比例
        """
        px = int(u * w)
        py = int(v * h)
        # 边界保护
        px = np.clip(px, 0, w-1)
        py = np.clip(py, 0, h-1)
        
        # 获取深度值 (0-255)
        d_val = depth_img[py, px]
        
        # 缩放算法：
        # 假设 Depth 255 (最白) 是相机近平面，缩放为 1.0
        # 假设 Depth 0 (最黑) 是无穷远，缩放为 0.0
        # 这里的指数 1.0 是线性关系，你可以根据效果调整为 1.2 或 0.8
        scale_factor = (d_val / 255.0) ** 1.0 
        
        # 设置最小缩放，防止物体在远处消失 (例如最小 0.1 倍)
        scale_factor = max(scale_factor, 0.1)
        
        return scale_factor * base_scale, d_val

    # ==========================
    # Part 3: 生成 JSON 数据
    # ==========================
    
    scene_data = {
        "scene_name": os.path.basename(rgb_path),
        "resolution": [w, h],
        "lighting": {
            "sun_position": {
                "pixel": [int(sun_x), int(sun_y)],
                "uv": [round(sun_x/w, 4), round(sun_y/h, 4)],
                # 将 UV 映射到 Unity Skybox Rotation (0-360度)
                # Unity Skybox 旋转通常对应 U 轴
                "rotation_angle": round((sun_x/w) * 360, 2)
            },
            "sun_intensity_estimate": round(max_val / 255.0, 2),
            "ambient_color_rgb": ambient_rgb
        },
        # 预计算几个参考点的缩放比例 (例如地面、中间、天空)
        "reference_scales": {
            "center": get_scale_at(0.5, 0.5)[0],
            "bottom_ground": get_scale_at(0.5, 0.8)[0], # 通常放置 Avatar 的位置
        }
    }

    # ==========================
    # 可视化输出 (Optional)
    # ==========================
    # 在图上画个圈标记太阳
    debug_img = rgb_img.copy()
    cv2.circle(debug_img, (sun_x, sun_y), 50, (0, 0, 255), 5)
    cv2.putText(debug_img, "SUN", (sun_x+60, sun_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_path = os.path.join(output_dir, "scene_meta.json")
    vis_path = os.path.join(output_dir, "scene_debug.jpg")
    
    with open(json_path, 'w') as f:
        json.dump(scene_data, f, indent=4)
        
    cv2.imwrite(vis_path, debug_img)
    
    print(f"✅ JSON Saved: {json_path}")
    print(f"✅ Debug Image: {vis_path}")
    
    # 打印测试：假设我们在地面放置 Avatar (UV: 0.5, 0.75)
    test_u, test_v = 0.5, 0.75
    scale, d_val = get_scale_at(test_u, test_v)
    print(f"\n🎯 Avatar Placement Test at UV({test_u}, {test_v}):")
    print(f"   - Depth Value: {d_val}/255")
    print(f"   - Rec. Scale : {scale:.2f}x (Based on depth)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True, help='Path to RGB panorama')
    parser.add_argument('--depth', required=True, help='Path to Depth panorama')
    parser.add_argument('--out', default='./out', help='Output directory')
    
    args = parser.parse_args()
    analyze_scene(args.rgb, args.depth, args.out)