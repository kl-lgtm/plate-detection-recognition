import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
# 路径基于脚本所在目录：开源部署时将 dataset 目录放在脚本同级
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = str(BASE_DIR / "dataset" / "ccpd_base_199996")  # 原始数据集路径
REDUCED_DATA_PATH = str(BASE_DIR / "dataset" / "ccpd_selected_15k")  # 缩减后的数据集路径
SAMPLE_NUM = 15000  # 目标样本数
RANDOM_SEED = 42  # 固定随机种子，结果可复现
# ----------------------------------------------------------------

# 创建缩减后的数据目录（已存在则不报错，避免覆盖原始数据）
os.makedirs(REDUCED_DATA_PATH, exist_ok=True)

# 获取所有原始图像文件名（仅保留.jpg，过滤其他无关文件）
all_img_files = [f for f in os.listdir(RAW_DATA_PATH) if f.lower().endswith(".jpg")]
print(f"✅ 原始样本总数：{len(all_img_files)}张")
print(f"✅ 开始随机无放回抽取{SAMPLE_NUM}张样本...")

# 固定随机种子，保证每次抽样结果完全一致，支持复现
random.seed(RANDOM_SEED)
# 随机无放回抽样，不会重复抽取同一张图片
sampled_files = random.sample(all_img_files, SAMPLE_NUM)

# 批量复制抽样后的【图片+标签】，并显示进度条
success_count = 0  # 成功复制的样本数
for img_file in tqdm(sampled_files, desc="📄 样本复制进度", ncols=80):
    # 拼接图片的源路径和目标路径
    img_src_path = os.path.join(RAW_DATA_PATH, img_file)
    img_dst_path = os.path.join(REDUCED_DATA_PATH, img_file)

    # 拼接 同名txt标签 的源路径和目标路径【核心补充】CCPD数据集必备
    txt_file = img_file.replace(".jpg", ".txt")
    txt_src_path = os.path.join(RAW_DATA_PATH, txt_file)
    txt_dst_path = os.path.join(REDUCED_DATA_PATH, txt_file)

    try:
        # 复制图片文件
        shutil.copy(img_src_path, img_dst_path)
        # 复制对应的标签文件（如果存在的话，CCPD数据集一定存在）
        if os.path.exists(txt_src_path):
            shutil.copy(txt_src_path, txt_dst_path)
        success_count += 1
    except Exception as e:
        # 异常捕获：遇到损坏文件/权限问题时，跳过并打印错误，不中断程序
        print(f"\n❌ 跳过损坏文件 {img_file}，错误信息：{str(e)}")

# 验证最终结果：统计真实的有效样本数（只统计jpg）
final_img_num = len([f for f in os.listdir(REDUCED_DATA_PATH) if f.lower().endswith(".jpg")])
print(f"\n✅ 样本缩减完成！")
print(f"✅ 成功复制样本数：{success_count} 张")
print(f"✅ 目标文件夹最终样本数：{final_img_num} 张")
print(f"✅ 缩减后数据集路径：{REDUCED_DATA_PATH}")