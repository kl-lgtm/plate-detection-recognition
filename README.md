# 基于 YOLOv5 与 PaddleOCR 的车牌自动检测与识别系统

基于轻量 YOLOv5n 目标检测与 PaddleOCR 文字识别的中文车牌自动检测与识别系统，覆盖数据集筛选、标注转换、模型训练（CPU 环境可跑）、检测识别到结果导出的完整流程。提供可视化桌面界面，支持单张/文件夹批量识别、结果表格展示与 CSV/Excel 导出。基于 CCPD 公开数据集 15000 张样本训练，验证集 mAP@0.5 达 0.995，2250 张测试集实测检测成功率 96.98%、字符识别准确率≥80%，识别置信度均值 0.822。

## 功能特性

- 车牌目标检测（YOLOv5n，轻量模型，CPU 可训练/推理）
- 中文车牌字符识别（PaddleOCR，车牌字符白名单 + 格式校验）
- 单张识别 / 文件夹批量识别
- 可视化桌面界面（拖拽图片、暂停/恢复、实时表格展示）
- 识别结果导出 CSV / Excel

## 技术栈

Python、YOLOv5（ultralytics）、PaddleOCR、OpenCV、Tkinter、Pandas

## 系统流程

```text
CCPD 公开数据集 (199996 张)
        ↓ select_ccpd.py        随机抽取 15000 张
        ↓ split_ccpd_selected_15k.py   按 7:1.5:1.5 划分 train/val/test
        ↓ convert_ccpd_to_yolo.py      转为 YOLO 标注格式
        ↓ train_plate.py        yolov5n 训练（CPU）
        ↓ best.pt
        ↓ PlateRecognitionUI.py 检测 + OCR 识别 + 结果导出
```

## 核心指标（测试集 2250 张实测）

| 指标 | 数值 |
| --- | --- |
| 验证集 mAP@0.5 | **0.995** |
| 车牌检测成功率 | **96.98%**（2182/2250） |
| 字符识别准确率 | **≥80%** |
| 识别置信度均值 | **0.822** |

> 指标来源：训练日志 `runs/train/exp/results.csv` 与测试结果表 `plate_detection_result.xlsx`。

## 运行步骤

### 1. 准备数据集

从 [CCPD 数据集](https://github.com/detectRecog/CCPD) 下载 `ccpd_base`，放入 `dataset/ccpd_base_199996/`：

```bash
# 1) 随机抽取 15000 张
python select_ccpd.py

# 2) 划分 train/val/test（7:1.5:1.5）
python split_ccpd_selected_15k.py

# 3) 转换为 YOLO 标注格式
python convert_ccpd_to_yolo.py
```

### 2. 训练

确认 `data.yaml` 与脚本同目录（含训练/验证/测试图片路径），执行：

```bash
python train_plate.py
```

训练产出 `runs/train/exp/weights/best.pt`。

### 3. 识别

```bash
python PlateRecognitionUI.py
```

打开界面后支持拖拽单张图片或选择文件夹批量识别。

## 目录结构

```text
plate-detection-recognition/
├─ README.md
├─ select_ccpd.py
├─ split_ccpd_selected_15k.py
├─ convert_ccpd_to_yolo.py
├─ train_plate.py
├─ PlateRecognitionUI.py
├─ data.yaml              # 训练配置（需自行准备）
├─ dataset/               # 数据集（公开数据，自行下载，不入库）
└─ runs/                  # 训练产物（权重不入库）
```

## 说明与注意事项

- **模型权重不随仓库分发**：`best.pt` / `yolov5n.pt` 体积较大，请按上述步骤自行训练获得，或联系作者获取。
- 所有脚本路径基于脚本所在目录（`BASE_DIR`），请保持 `dataset/`、`data.yaml` 与脚本同级。
- PaddleOCR 识别使用车牌字符白名单，支持常见省份简称与蓝/黄牌字符。
- 本项目为课程设计性质的个人学习项目，数据来自公开数据集，仅用于学习交流。

## License

仅供学习交流使用。
