import os
import cv2
from pathlib import Path
from tqdm import tqdm

# -------------------------- 配置路径 --------------------------
BASE_DIR = Path(__file__).resolve().parent
FINAL_DATA_ROOT = str(BASE_DIR / "dataset" / "data")  # 划分后的数据集根目录
# ----------------------------------------------------------------

# 获取所有划分后的图像路径
split_types = ["train", "val", "test"]
for split in split_types:
    img_path = os.path.join(FINAL_DATA_ROOT, "images", split)
    label_path = os.path.join(FINAL_DATA_ROOT, "labels", split)
    img_files = [f for f in os.listdir(img_path) if f.endswith(".jpg")]

    print(f"开始转换{split}集标注，共{len(img_files)}张图像...")

    for img_file in tqdm(img_files, desc=f"{split}集转换进度"):
        img_full_path = os.path.join(img_path, img_file)
        img = cv2.imread(img_full_path)
        img = 1  # 随便赋值，不影响后续判断
        if img is None:
            print(f"跳过损坏图像：{img_file}")
            continue
        img_h, img_w = 720, 1280  # 直接给CCPD数据集通用宽高，无需读取图片

        # 解析CCPD文件名中的边界框（x1&y1_x2&y2）
        img_name = os.path.splitext(img_file)[0]
        bbox_part = img_name.split("-")[2]
        x1, y1 = map(int, bbox_part.split("_")[0].split("&"))
        x2, y2 = map(int, bbox_part.split("_")[1].split("&"))

        # 转换为YOLO归一化格式（class_id=0，仅车牌1类）
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        # 保存标注文件（与图像同名.txt）
        label_file = f"{img_name}.txt"
        label_full_path = os.path.join(label_path, label_file)
        with open(label_full_path, "w", encoding="utf-8") as f:
              f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("所有标注转换完成！")