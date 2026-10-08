# -*- coding: utf-8 -*-
# YOLOv5 车牌检测训练代码【断点续训正式版 | 修复列名+格式匹配】
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')  # 关闭无关警告，终端日志更干净

# ======================== 仅需确认这1处路径正确 ========================
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML_PATH = str(BASE_DIR / "data.yaml")
RESUME_TRAIN = True   # 已开启断点续训（核心开关）
# ======================================================================

# ======================== CPU专属优化配置（无需改动）================
DEVICE = "cpu"
BATCH_SIZE = 8
EPOCHS = 21
IMG_SIZE = 640
MODEL_TYPE = "yolov5n"
LOG_DIR = "runs/train/exp"
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)
# ======================================================================

def train_license_plate_detector():
    print("="*70)
    print("✅ 启动车牌检测训练【CPU模式 | 断点续训开启 | 打印完整训练过程】")
    print(f"📊 训练配置：批次{BATCH_SIZE} | 总轮数{EPOCHS} | 图像尺寸{IMG_SIZE}")
    print(f"📁 数据集配置：{DATA_YAML_PATH}")
    print(f"🔄 断点续训：{'开启' if RESUME_TRAIN else '关闭'}")
    print(f"📈 训练过程会实时打印损失、精度等指标（verbose=True）！")
    print("="*70)

    # 1. 校验data.yaml文件是否存在（必过校验）
    if not os.path.exists(DATA_YAML_PATH):
        print(f"\n❌ 致命错误：找不到data.yaml配置文件！")
        print(f"⚠️  当前路径：{DATA_YAML_PATH}")
        return

    # 2. 加载模型（断点续训逻辑：优先加载last.pt）
    if RESUME_TRAIN:
        last_weight = os.path.join(LOG_DIR, "weights/last.pt")
        if os.path.exists(last_weight):
            model = YOLO(last_weight)
            print(f"\n✅ 断点续训成功！加载权重：{last_weight}")
            print(f"✅ 训练将从上次中断的轮数继续，直到完成{EPOCHS}轮！")
        else:
            print(f"\n⚠️  未找到断点权重 {last_weight}，自动切换为从头训练！")
            model = YOLO(f"{MODEL_TYPE}.yaml").load(f"{MODEL_TYPE}.pt")
    else:
        model = YOLO(f"{MODEL_TYPE}.yaml").load(f"{MODEL_TYPE}.pt")
        print(f"\n✅ 成功加载预训练模型，准备从头训练！")

    # 3. 核心训练逻辑（断点续训+实时打印训练过程）
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=0,               # CPU必设0，避免内存溢出
        cache="disk",            # 缓存数据到磁盘，加速加载
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        save=True,               # 自动保存best.pt和last.pt
        save_period=1,           # 每轮保存last.pt，确保断点不丢失
        val=True,                # 每轮验证，生成mAP精度指标
        plots=True,              # 自动生成官方可视化图表
        exist_ok=True,           # 覆盖旧日志，支持断点续训
        project="runs/train",
        name="exp",
        verbose=True,            # 关键：实时打印每轮训练损失、精度！
    )

    # 4. 训练完成后绘图（修复列名：train/box → 匹配你的results.csv）
    print("\n✅ 训练完成！开始生成训练曲线...")
    csv_path = os.path.join(LOG_DIR, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(14, 6), dpi=300)

        # 子图1：损失曲线（修复列名：train/box、val/box 替代 train/box_loss、val/box_loss）
        plt.subplot(1, 2, 1)
        plt.plot(df["epoch"], df["train/box"], label="训练集-边界框损失", color="#e74c3c", linewidth=2)
        plt.plot(df["epoch"], df["val/box"], label="验证集-边界框损失", color="#3498db", linewidth=2)
        plt.title("车牌检测 - 边界框损失曲线", fontsize=12, fontweight="bold")
        plt.xlabel("训练轮数 (Epoch)")
        plt.ylabel("损失值")
        plt.legend()
        plt.grid(alpha=0.3, linestyle="--")

        # 子图2：精度曲线（修复列名：metrics/mAP50 替代 metrics/mAP50(B)）
        plt.subplot(1, 2, 2)
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP@0.5 检测精度", color="#2ecc71", linewidth=2)
        plt.title("车牌检测 - 检测精度曲线", fontsize=12, fontweight="bold")
        plt.xlabel("训练轮数 (Epoch)")
        plt.ylabel("mAP@0.5 精度值")
        plt.legend()
        plt.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig("loss_curve.png", bbox_inches="tight")
        plt.close()
        print("✅ 训练曲线已保存为 loss_curve.png！")

    # 5. 训练结果汇总（断点续训版）
    print("\n" + "="*70)
    print("🎉 断点续训完成！训练过程已完整打印！")
    print(f"📌 训练日志：终端实时输出 + {LOG_DIR}/results.csv")
    print(f"📌 最优模型：{LOG_DIR}/weights/best.pt（精度最高）")
    print(f"📌 断点权重：{LOG_DIR}/weights/last.pt（下次续训直接用）")
    print(f"📌 可视化图表：{LOG_DIR}/results.png + 根目录loss_curve.png")
    print("="*70)

if __name__ == "__main__":
    train_license_plate_detector()