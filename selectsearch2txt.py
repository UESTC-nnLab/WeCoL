# -*- coding: utf-8 -*-
import os
import numpy as np
import skimage.io
from selectivesearch import selective_search
from tqdm import tqdm

# ====================== 参数设置 ======================
IMAGE_LIST_TXT = "/home/dww/OD/weak_stream5/pseudo_gt/pseudo_gt_temp/coco_train_DAUB.txt"
OUTPUT_TXT = "pseudo_DAUB.txt"  # 输出TXT文件名

# Selective Search 参数
SCALE = 200
SIGMA = 0.9
MIN_SIZE = 5

# 后处理参数
MAX_BOX_AREA = 150        # 过滤面积过大的候选框
MIN_BOX_AREA = 30         # 过滤面积过小的噪声
ASPECT_RATIO_TH = 4.0     # 过滤极端长宽比的候选框

# ====================== 1. 加载图像路径列表 ======================
def load_image_paths(txt_path):
    """加载并验证图像路径"""
    with open(txt_path, "r") as f:
        lines = [line.strip().split(" ")[0] for line in f.readlines()]
    
    valid_paths = []
    for path in lines:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"警告：跳过不存在的路径 {path}")
    
    print(f"成功加载 {len(valid_paths)} 张图像")
    return valid_paths

image_paths = load_image_paths(IMAGE_LIST_TXT)

# ====================== 2. 处理单张图像的核心函数 ======================
def process_single_image(img_path):
    """生成候选框并过滤"""
    # 加载图像
    img = skimage.io.imread(img_path)
    H, W = img.shape[0], img.shape[1]
    
    # 处理图像通道
    if img.ndim == 2:
        img = np.stack([img]*3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 运行 Selective Search
    img_uint8 = img.astype(np.uint8)
    _, regions = selective_search(img_uint8, scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE)
    
    # 提取候选框坐标
    candidate_boxes = []
    for region in regions:
        x, y, w, h = region["rect"]
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(W-1, x + w))
        y2 = int(min(H-1, y + h))
        if x2 > x1 and y2 > y1:
            candidate_boxes.append([x1, y1, x2, y2])
    
    if not candidate_boxes:
        return []
    
    # 转换为 NumPy 数组
    boxes = np.array(candidate_boxes, dtype=np.int32)
    
    # 计算几何特征
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    aspect_ratios = np.maximum(widths / (heights + 1e-5), heights / (widths + 1e-5))
    
    # 复合过滤条件
    keep = (areas >= MIN_BOX_AREA) & (areas <= MAX_BOX_AREA) & (aspect_ratios <= ASPECT_RATIO_TH)
    return boxes[keep].tolist()

# ====================== 3. 批量处理并保存结果 ======================
with open(OUTPUT_TXT, "w") as f_out:
    for img_path in tqdm(image_paths, desc="生成伪标签"):
        try:
            # 生成候选框
            boxes = process_single_image(img_path)
            if not boxes:
                continue
                
            # 转换为字符串格式
            box_strs = [f"{x1},{y1},{x2},{y2},0" for (x1, y1, x2, y2) in boxes]
            
            # 写入文件
            line = f"{img_path} {' '.join(box_strs)}\n"
            f_out.write(line)
            
        except Exception as e:
            print(f"处理失败: {img_path}，错误: {str(e)}")
            continue

print(f"处理完成！结果已保存至 {OUTPUT_TXT}")