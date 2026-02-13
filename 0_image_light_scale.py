# 切换到独立环境
# python3 -m venv venv
# source venv/bin/activate
# B. 前端如何使用 Scale Grid？
# 这份代码生成的 JSON 包含了一个结构化的网格数据。在前端（Three.js / React / Vue）中实现小孩跑动逻辑非常简单：
# 加载 JSON：将 nav_mesh.points 存入一个二维数组 Grid[row][col]。
# 获取 Avatar 当前坐标：假设小孩跑到了 u = 0.55, v = 0.82。
# 查找最近网格点：u=0.55 介于网格列 19 和 20 之间。v=0.82 介于网格行 6 和 7 之间。
# 计算 Scale：找到这 4 个相邻点的 scale 值。
# 使用简单的双线性插值 (Bilinear Interpolation) 算出当前点的精确 Scale。
# 公式：$Scale = w_1 S_{TL} + w_2 S_{TR} + w_3 S_{BL} + w_4 S_{BR}$
import cv2
import numpy as np
import json
import argparse
import os

def analyze_scene_advanced(rgb_path, depth_path, output_dir):
    # 1. 读取图像
    rgb_img = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) # 单通道深度
    
    if rgb_img is None or depth_img is None:
        print("❌ Error: 无法读取 RGB 或 Depth 图片")
        return

    h, w = rgb_img.shape[:2]
    print(f"🖼  Processing: {w}x{h}")

    # ==========================================
    # Part 1: 高级多光源检测 (Multi-Light Detection)
    # ==========================================
    
    # 转换为灰度
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # 策略：不进行大范围模糊，保留锐利的高光点
    # 仅做极微小的模糊以消除噪点
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 阈值化：只提取极亮区域 (亮度 > 240/255)
    # 这能有效过滤掉普通的白云，只留下太阳或路灯核心
    ret, thresh = cv2.threshold(gray_blur, 240, 255, cv2.THRESH_BINARY)
    
    # 连通域分析：找出所有独立的发光块
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    light_sources = []
    
    # 遍历所有连通域 (label 0 是背景，跳过)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 过滤掉太小的噪点 (例如只有 1-2 个像素的亮点)
        if area < 5: 
            continue
            
        # 获取该区域的中心坐标
        cx, cy = centroids[i]
        
        # 获取该区域内的最大亮度 (在原始灰度图上找，而不是二值图)
        # 创建一个掩码只提取当前光源区域
        mask = (labels == i).astype(np.uint8)
        min_val, max_val, _, max_loc = cv2.minMaxLoc(gray, mask=mask)
        
        # 计算综合评分：通常太阳是 最亮 且 相对集中
        # 这里我们主要按 max_intensity 排序，如果亮度一样，按面积排序
        score = max_val * 1000 + area 
        
        light_sources.append({
            "id": i,
            "type": "point_light",
            "score": float(score),
            "intensity": int(max_val), # 0-255
            "area": int(area),
            "pixel_coords": [int(cx), int(cy)],
            "uv": [round(cx/w, 4), round(cy/h, 4)],
            # 将 UV 映射到 360 度 (U=0.5 -> 180度)
            "angle_yaw": round((cx/w) * 360, 2),
            "angle_pitch": round((cy/h) * 180 - 90, 2) # -90(底) 到 90(顶)
        })
    
    # 按评分降序排列 (最可能是太阳的排第一)
    light_sources.sort(key=lambda x: x["score"], reverse=True)
    
    # 取前 5 个光源 (适应夜景多路灯情况)
    top_lights = light_sources[:5]

    # ==========================================
    # Part 2: 网格化缩放地图 (Grid Scale Map)
    # ==========================================
    
    # 配置网格密度
    # 仅覆盖下半部分 (Ground)
    GRID_ROWS = 10  # 垂直方向行数 (只取下半截)
    GRID_COLS = 36  # 水平方向列数 (每10度一个点)
    
    scale_points = []
    
    # 垂直方向：从 50% (地平线) 到 95% (脚下)
    # 避免 100% 极点，因为那里贴图扭曲极大
    row_steps = np.linspace(0.55, 0.95, GRID_ROWS)
    col_steps = np.linspace(0.0, 1.0, GRID_COLS, endpoint=False) # 0-360度
    
    for r_idx, v_ratio in enumerate(row_steps):
        for c_idx, u_ratio in enumerate(col_steps):
            
            px = int(u_ratio * w)
            py = int(v_ratio * h)
            
            # 边界保护
            px = int(np.clip(px, 0, w-1))
            py = int(np.clip(py, 0, h-1))
            
            # --- 深度采样优化 ---
            # 不要只取单点像素，取 5x5 区域平均值，防止踩到噪点
            patch_size = 5
            y1 = max(0, py - patch_size // 2)
            y2 = min(h, py + patch_size // 2 + 1)
            x1 = max(0, px - patch_size // 2)
            x2 = min(w, px + patch_size // 2 + 1)
            
            depth_patch = depth_img[y1:y2, x1:x2]
            avg_depth = np.mean(depth_patch)

            # ==========================================
            # 🚀 核心修正：基于全景透视的几何计算
            # ==========================================
            
            # 1. 几何对地距离估算 (Geometric Ground Scale)
            # 全景图中，V=0.5 是地平线(无穷远)，V=1.0 是脚下(最近)
            # 角度 alpha = (v - 0.5) * PI (从地平线向下看的角度)
            # 物理距离 D = CameraHeight / tan(alpha)
            # 缩放比例 Scale ∝ 1 / D ∝ tan(alpha)
            
            # 为了防止 V=0.5 时 tan(0)=0 导致消失，我们设置一个最小角度偏移
            v_clamped = max(v_ratio, 0.52) # 0.52 约等于向下看 3.6度，保证远处不为0
            
            # 计算正切值 (这就是符合物理的近大远小曲线)
            # 越靠近 1.0，tan 值增长越快
            geometric_factor = np.tan((v_clamped - 0.5) * np.pi)
            
            # 2. 深度图修正 (AI Depth Correction)
            # 我们主要信任几何计算，但是如果深度图显示这里突然变黑(变远)或变白(障碍物)
            # 我们用深度图做一个微调系数。
            # 归一化深度 (0.0 - 1.0)
            d_norm = avg_depth / 255.0
            
            # 混合策略：
            # 基础 Scale 完全由几何位置决定 (解决近处反而小的问题)
            # 深度图只负责微调 (比如雪地里有个坑，AI看出来了，深度变小，Scale略微变小)
            # 这里给几何权重 80%，AI 深度权重 20%
            
            # 经验系数：调节 global_scale_mult 让整体大小合适
            global_scale_mult = 3.5 
            
            # 最终公式：Scale = 几何曲线 * (0.5 + 0.5 * AI深度) * 全局系数
            # 这样即使 AI 在底部判断失误，几何曲线也能强制把 Scale 拉大
            final_scale = geometric_factor * (0.5 + 0.5 * d_norm) * global_scale_mult
            
            # 3. 限制极值 (防止脚底下无限大)
            final_scale = np.clip(final_scale, 0.1, 8.0)
            
            scale_points.append({
                "grid_pos": [c_idx, r_idx], # 网格索引，方便前端查找
                "uv": [round(u_ratio, 4), round(v_ratio, 4)],
                "pixel": [px, py],
                "depth_val": int(avg_depth),
                "scale": round(float(final_scale), 3)
            })

    # ==========================
    # Part 3: 输出与可视化
    # ==========================
    
    # 绘制 Debug 图片
    debug_img = rgb_img.copy()
    
    # 1. 画光源
    for i, light in enumerate(top_lights):
        cx, cy = light["pixel_coords"]
        # 第一名(太阳)用粗绿色圈，其他用细黄色圈
        color = (0, 255, 0) if i == 0 else (0, 255, 255) 
        thickness = 5 if i == 0 else 2
        radius = int(np.sqrt(light["area"])) + 20
        
        cv2.circle(debug_img, (cx, cy), radius, color, thickness)
        cv2.putText(debug_img, f"Light {i+1}", (cx+radius, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 2. 绘制缩放网格点 (重点修改部分)
    print("🎨 Drawing Scale Circles...")
    
    for pt in scale_points:
        px, py = pt["pixel"]
        scale = pt["scale"]
        
        # --- 核心修正逻辑 ---
        # 我们不能用固定的像素值，必须基于图片高度 (h) 计算。
        # 设定：在 Scale=1.0 时，圆圈半径是图片高度的 2% (大约是一个人的占地半径)
        # 例如 2880p 高度 -> 1.0 scale = 57 像素半径
        base_radius_ratio = 0.02 
        radius_px = int(h * base_radius_ratio * scale)
        
        # 确保最小可见性 (至少3个像素)
        radius_px = max(radius_px, 3)
        
        # A. 绘制红点 (脚底锚点) - 实心
        # 锚点大小也随分辨率变化，设为高度的 0.3%
        anchor_radius = max(int(h * 0.003), 2)
        cv2.circle(debug_img, (px, py), anchor_radius, (0, 0, 255), -1) 
        
        # B. 绘制蓝圈 (Avatar 缩放参考) - 空心
        # 线宽随分辨率变化
        line_thickness = max(int(h * 0.001), 1)
        cv2.circle(debug_img, (px, py), radius_px, (255, 200, 0), line_thickness) 

        # (可选) 每隔几个点标一下数值，防止太密集
        if pt["grid_pos"][0] % 4 == 0 and pt["grid_pos"][1] % 2 == 0:
            font_scale = h / 2000.0 # 字体随图片大小缩放
            cv2.putText(debug_img, f"{scale:.2f}", (px + 10, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    # 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_path = os.path.join(output_dir, "scene_meta_v3.json")
    vis_path = os.path.join(output_dir, "scene_debug_v3.jpg")
    
    output_data = {
        "scene_name": os.path.basename(rgb_path),
        "resolution": [w, h],
        "lights": top_lights,
        "nav_mesh": {
            "type": "grid",
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "points": scale_points
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    cv2.imwrite(vis_path, debug_img)
    
    print(f"✅ Analysis Complete.")
    print(f"   - Found {len(top_lights)} lights.")
    print(f"   - Generated {len(scale_points)} scale points.")
    print(f"   - JSON: {json_path}")
    print(f"   - Debug Img: {vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--depth', required=True)
    parser.add_argument('--out', default='./out')
    args = parser.parse_args()
    
    analyze_scene_advanced(args.rgb, args.depth, args.out)