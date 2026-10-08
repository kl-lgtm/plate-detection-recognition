import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------- 配置路径 --------------------------
BASE_DIR = Path(__file__).resolve().parent
REDUCED_DATA_PATH = str(BASE_DIR / "dataset" / "ccpd_selected_15k")  # 缩减后的数据集路径（1.5万张）
FINAL_DATA_ROOT = str(BASE_DIR / "dataset" / "data")  # 最终划分后的数据集根目录
# ----------------------------------------------------------------

# 最终数据集目录结构（自动创建）
img_train_path = os.path.join(FINAL_DATA_ROOT, "images", "train")
img_val_path = os.path.join(FINAL_DATA_ROOT, "images", "val")
img_test_path = os.path.join(FINAL_DATA_ROOT, "images", "test")
label_train_path = os.path.join(FINAL_DATA_ROOT, "labels", "train")
label_val_path = os.path.join(FINAL_DATA_ROOT, "labels", "val")
label_test_path = os.path.join(FINAL_DATA_ROOT, "labels", "test")
img_test_path = os.path.join(FINAL_DATA_ROOT, "images", "test")

for path in [img_train_path, img_val_path, img_test_path, label_train_path, label_val_path, label_test_path]:
    os.makedirs(path, exist_ok=True)

# 获取缩减后所有样本名称（不含后缀）
sample_names = [os.path.splitext(f)[0] for f in os.listdir(REDUCED_DATA_PATH) if f.endswith(".jpg")]
print(f"待划分样本总数：{len(sample_names)}张")

# 第一步：划分训练+验证集（85%）和测试集（15%）
train_val_names, test_names = train_test_split(
    sample_names, test_size=0.15, random_state=42, shuffle=True
)

# 第二步：划分训练集（70%总数据）和验证集（15%总数据）
train_names, val_names = train_test_split(
    train_val_names, test_size=0.176, random_state=42, shuffle=True  # 0.15/0.85≈0.176
)

print(f"划分结果：")
print(f"训练集：{len(train_names)}张 | 验证集：{len(val_names)}张 | 测试集：{len(test_names)}张")


# 定义复制函数（批量复制图像+后续生成的标注）
def copy_data(sample_list, split_type):
    """
    sample_list: 样本名称列表
    split_type: 划分类型（train/val/test）
    """
    # 图像目标路径和标注目标路径
    img_dst = os.path.join(FINAL_DATA_ROOT, "images", split_type)
    label_dst = os.path.join(FINAL_DATA_ROOT, "labels", split_type)

    # 复制图像
    for name in tqdm(sample_list, desc=f"复制{split_type}集图像"):
        src_img = os.path.join(REDUCED_DATA_PATH, f"{name}.jpg")
        dst_img = os.path.join(img_dst, f"{name}.jpg")

        shutil.copy(src_img, dst_img)


# 执行复制（先复制图像，标注在后续格式转换时生成）
copy_data(train_names, "train")
copy_data(val_names, "val")
copy_data(test_names, "test")

print("数据集划分完成！")
print(f"最终数据集路径：{FINAL_DATA_ROOT}")