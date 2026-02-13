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
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            
            # --- 深度采样优化 ---
            # 不要只取单点像素，取 5x5 区域平均值，防止踩到噪点
            patch_size = 5
            y1 = max(0, py - patch_size // 2)
            y2 = min(h, py + patch_size // 2 + 1)
            x1 = max(0, px - patch_size // 2)
            x2 = min(w, px + patch_size // 2 + 1)
            
            depth_patch = depth_img[y1:y2, x1:x2]
            if depth_patch.size == 0: continue
            avg_depth = np.mean(depth_patch)
            
            # --- 缩放算法 ---
            # 1. 深度基础缩放 (Depth Scale): 越白(255)越近，越大
            #    公式：(depth / 255) ^ gamma
            # 假设: Depth 255 (最近) -> Scale 2.5
            #       Depth 50  (远)   -> Scale 0.3
            # 你可以调节 gamma 指数来控制衰减速度
            base_scale = (avg_depth / 255.0) ** 1.0
            
            # 2. 投影修正 (Projection Correction):
            #    在等距柱状投影中，越靠近底部，像素被横向拉伸得越厉害。
            #    为了视觉补偿，通常越靠近底部物体应该稍微“扁/宽”一点，或者整体调大。
            #    这里做一个简单的线性补偿：越靠下(v接近1)，Scale 适当放大
            projection_factor = 1.0 + (v_ratio - 0.5) * 0.8
            
            final_scale = base_scale * projection_factor * 2.5 # 乘一个系数让整体数值好看
            final_scale = np.clip(final_scale, 0.1, 5.0) # 限制范围
            
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
        
    json_path = os.path.join(output_dir, "scene_meta_v2.json")
    vis_path = os.path.join(output_dir, "scene_debug_v2.jpg")
    
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