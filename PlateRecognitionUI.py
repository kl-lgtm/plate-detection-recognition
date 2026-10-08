import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="module compiled against ABI version")
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")

import os
import cv2
import time
import csv  # 新增：用于保存表格
import numpy as np
import threading
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import Tk, Frame, Label, Button, Entry, StringVar, messagebox, filedialog, DoubleVar, Scale, Scrollbar
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import ttk  # 表格组件
from paddleocr import PaddleOCR

# ======================== 核心配置 ========================
BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = str(BASE_DIR / "runs/train/exp/weights/best.pt")
OUTPUT_PATH = str(BASE_DIR / "result_images")
TABLE_SAVE_PATH = str(BASE_DIR)  # 表格保存到项目根目录
CONF_THRESH = 0.5  # YOLO检测置信度阈值
DEVICE = "cpu"
# 车牌专属字符白名单
PLATE_CHAR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ苏浙皖沪京津渝冀晋蒙辽吉黑闽赣鲁豫鄂湘粤桂琼川贵云陕甘青宁新藏·"
# 中文字体路径（Windows默认，其他系统请修改）
FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"  # 黑体


# ==========================================================


# ======================== 车牌格式轻量校验 ========================
def check_plate_format(plate_char):
    plate_char = ''.join([c for c in plate_char if c in PLATE_CHAR_ALLOWLIST])
    if '·' not in plate_char and len(plate_char) >= 6:
        plate_char = plate_char[:2] + '·' + plate_char[2:]
    return plate_char


# ======================== 加载模型（简化OCR配置） ========================
def load_models():
    # 加载YOLO
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ 成功加载YOLOv5模型：{YOLO_MODEL_PATH}")
    except Exception as e:
        messagebox.showerror("YOLO加载失败", f"错误信息：\n{str(e)}")
        exit()

    # 加载PaddleOCR（移除角度分类，适配车牌）
    try:
        plate_ocr = PaddleOCR(
            use_angle_cls=False,  # 车牌默认水平，取消角度分类
            lang="ch",
            use_gpu=False,
            rec_char_dict_path=None,
            rec_algorithm="SVTR_LCNet",
            rec_allow_list=PLATE_CHAR_ALLOWLIST,
            rec_image_shape="3, 64, 256",  # 车牌最优识别尺寸
            rec_batch_num=4
        )
        print(f"✅ 成功加载PaddleOCR车牌识别模型")
        return yolo_model, plate_ocr
    except Exception as e:
        messagebox.showerror("OCR加载失败", f"错误信息：\n{str(e)}")
        exit()


# ======================== 核心功能：用PIL绘制中文，彻底解决???问题 ========================
def detect_recognize(img_path, yolo_model, plate_ocr, conf_thresh):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, "识别失败", 0.0, []

    # 加载中文字体（PIL专用，100%兼容）
    try:
        font = ImageFont.truetype(FONT_PATH, 40)
        font_loaded = True
    except Exception as e:
        print(f"⚠️ 加载中文字体失败：{e}")
        print(f"⚠️ 请检查字体路径是否正确：{FONT_PATH}")
        font_loaded = False

    results = yolo_model(img, device=DEVICE, conf=conf_thresh)
    result_img = img.copy()
    plate_char = "识别失败"
    conf = 0.0
    coord = []

    for r in results:
        boxes = r.boxes.data.numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            box_conf = box[4]
            coord = [x1, y1, x2, y2]
            conf = box_conf

            # 裁剪车牌（直接用彩色图）
            plate_img = img[y1:y2, x1:x2]
            plate_img = cv2.resize(plate_img, (256, 64))

            # OCR识别（已正确识别中文+符号）
            try:
                ocr_result = plate_ocr.ocr(plate_img)
                if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                    plate_char = ocr_result[0][0][1][0].strip()
                    plate_char = check_plate_format(plate_char)
            except Exception as e:
                print(f"OCR识别错误：{e}")
                plate_char = "识别失败"

            # ========== 1. 加粗车牌框（OpenCV绘制） ==========
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # ========== 2. 独立文字背景框（OpenCV绘制） ==========
            text_bg_width = 400
            text_bg_height = 80
            text_bg_x1 = x1 - (text_bg_width - (x2 - x1)) // 2
            text_bg_x1 = max(0, text_bg_x1)
            text_bg_x2 = text_bg_x1 + text_bg_width
            text_bg_y1 = y1 - text_bg_height
            text_bg_y2 = y1
            # 绘制白色背景框
            cv2.rectangle(result_img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1)

            # ========== 3. 用PIL绘制中文文字 ==========
            text = f"{plate_char} ({conf:.2f})"
            if font_loaded:
                pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = text_bg_x1 + (text_bg_width - text_width) // 2
                text_y = text_bg_y1 + (text_bg_height - text_height) // 2
                draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
                result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                               1.6, 4)
                text_x = text_bg_x1 + (text_bg_width - text_width) // 2
                text_y = text_bg_y1 + (text_bg_height + text_height) // 2
                cv2.putText(result_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, (0, 0, 0), 4)

    return img, result_img, plate_char, conf, coord


# ======================== GUI界面：新增“导出表格”功能 ========================
class PlateRecognitionGUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("基于YOLOv5和PaddleOCR的车牌自动检测与识别系统")
        self.geometry("1500x800")
        self.resizable(True, True)

        # 核心变量
        self.img_path = None
        self.raw_img = None
        self.result_img = None
        self.yolo_model, self.plate_ocr = load_models()
        self.detect_count = 0
        self.batch_running = False
        self.pause_event = threading.Event()
        self.pause_event.set()

        # 创建结果保存文件夹
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

        # 初始化界面
        self._init_widgets()

    def _init_widgets(self):
        self.option_add("*Font", "微软雅黑 12")

        # 1. 顶部标题栏
        title_frame = Frame(self, bg="#87CEFA", height=70)
        title_frame.pack(fill="x", padx=10, pady=2)
        title_frame.pack_propagate(False)
        Label(title_frame, text="基于深度学习的车牌自动检测与识别系统",
              font=("微软雅黑", 20, "bold"), fg="#333333", bg="#87CEFA").pack(side="left", padx=20, pady=15)

        # ========== 主体：左右分栏（各占1/2） ==========
        main_frame = Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=2)

        # ---------------------- 左栏：用grid固定区域 ----------------------
        left_frame = Frame(main_frame, relief="solid", bd=2)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=0)
        left_frame.grid_rowconfigure(2, weight=0)
        left_frame.grid_columnconfigure(0, weight=1)

        # 左栏-row0：图片展示区
        self.img_display_frame = Frame(left_frame, relief="solid", bd=2)
        self.img_display_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.img_label = Label(self.img_display_frame, text="请拖拽图片或点击下方选择文件/文件夹",
                               font=("微软雅黑", 16), fg="#999999", bg="#f5f5f5")
        self.img_label.pack(fill="both", expand=True, padx=20, pady=20)
        self.img_label.drop_target_register(DND_FILES)
        self.img_label.dnd_bind('<<Drop>>', self._on_img_drop)

        # 左栏-row1：检测结果区
        result_frame = Frame(left_frame, padx=10, pady=5)
        result_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        Label(result_frame, text="检测结果", font=("微软雅黑", 14, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                 sticky="w", pady=3)

        Label(result_frame, text="识别结果：", font=("微软雅黑", 12)).grid(row=1, column=0, sticky="w", pady=1)
        self.plate_char_var = StringVar(value="——")
        Label(result_frame, textvariable=self.plate_char_var, font=("微软雅黑", 14, "bold"), fg="red").grid(row=1,
                                                                                                            column=1,
                                                                                                            sticky="w",
                                                                                                            padx=5)

        Label(result_frame, text="置信度：", font=("微软雅黑", 12)).grid(row=2, column=0, sticky="w", pady=1)
        self.conf_var = StringVar(value="0.00")
        Label(result_frame, textvariable=self.conf_var, font=("微软雅黑", 14, "bold"), fg="red").grid(row=2, column=1,
                                                                                                      sticky="w",
                                                                                                      padx=5)

        Label(result_frame, text="目标位置：", font=("微软雅黑", 12)).grid(row=3, column=0, columnspan=2, sticky="w",
                                                                          pady=3)
        Label(result_frame, text="xmin：", font=("微软雅黑", 10)).grid(row=4, column=0, sticky="w")
        self.xmin_var = StringVar(value="——")
        Label(result_frame, textvariable=self.xmin_var, font=("微软雅黑", 10)).grid(row=4, column=1, sticky="w", padx=5)
        Label(result_frame, text="ymin：", font=("微软雅黑", 10)).grid(row=5, column=0, sticky="w")
        self.ymin_var = StringVar(value="——")
        Label(result_frame, textvariable=self.ymin_var, font=("微软雅黑", 10)).grid(row=5, column=1, sticky="w", padx=5)
        Label(result_frame, text="xmax：", font=("微软雅黑", 10)).grid(row=6, column=0, sticky="w")
        self.xmax_var = StringVar(value="——")
        Label(result_frame, textvariable=self.xmax_var, font=("微软雅黑", 10)).grid(row=6, column=1, sticky="w", padx=5)
        Label(result_frame, text="ymax：", font=("微软雅黑", 10)).grid(row=7, column=0, sticky="w")
        self.ymax_var = StringVar(value="——")
        Label(result_frame, textvariable=self.ymax_var, font=("微软雅黑", 10)).grid(row=7, column=1, sticky="w", padx=5)

        # 左栏-row2：按钮区（新增“导出表格”按钮）
        op_frame = Frame(left_frame, padx=5, pady=3)
        op_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)

        btn_row1 = Frame(op_frame)
        btn_row1.pack(side="top", fill="x")
        Button(btn_row1, text="选择图片", font=("微软雅黑", 11, "bold"), bg="#2196F3", fg="white",
               command=self._select_single_img, width=12).pack(side="left", padx=2, pady=1)
        Button(btn_row1, text="选择文件夹", font=("微软雅黑", 11, "bold"), bg="#FF9800", fg="white",
               command=self._select_folder, width=12).pack(side="left", padx=2, pady=1)
        Button(btn_row1, text="暂停检测", font=("微软雅黑", 11, "bold"), bg="#FFC107", fg="black",
               command=self._pause_detect, width=12).pack(side="left", padx=2, pady=1)

        btn_row2 = Frame(op_frame)
        btn_row2.pack(side="top", fill="x")
        Button(btn_row2, text="保存结果", font=("微软雅黑", 11, "bold"), bg="#4CAF50", fg="white",
               command=self._save_result, width=12).pack(side="left", padx=2, pady=1)
        Button(btn_row2, text="清空表格", font=("微软雅黑", 11, "bold"), bg="#f44336", fg="white",
               command=self._clear_table, width=12).pack(side="left", padx=2, pady=1)
        # 新增：导出表格按钮
        Button(btn_row2, text="导出表格", font=("微软雅黑", 11, "bold"), bg="#9C27B0", fg="white",
               command=self._export_table, width=12).pack(side="left", padx=2, pady=1)
        Button(btn_row2, text="退出系统", font=("微软雅黑", 11, "bold"), bg="#607D8B", fg="white",
               command=self.quit, width=12).pack(side="left", padx=2, pady=1)

        # ---------------------- 右栏：保持不变 ----------------------
        right_frame = Frame(main_frame, relief="solid", bd=2)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        Label(right_frame, text="检测与识别结果", font=("微软雅黑", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        style = ttk.Style()
        style.configure("Treeview", font=("微软雅黑", 12), rowheight=25)
        style.configure("Treeview.Heading", font=("微软雅黑", 12, "bold"))

        self.result_table = ttk.Treeview(right_frame, columns=("序号", "文件路径", "识别结果", "置信度", "坐标位置"),
                                         show="headings", style="Treeview")
        self.result_table.heading("序号", text="序号")
        self.result_table.heading("文件路径", text="文件路径")
        self.result_table.heading("识别结果", text="识别结果")
        self.result_table.heading("置信度", text="置信度")
        self.result_table.heading("坐标位置", text="坐标位置")
        self.result_table.column("序号", width=60, anchor="center")
        self.result_table.column("文件路径", width=350, anchor="w")
        self.result_table.column("识别结果", width=120, anchor="center")
        self.result_table.column("置信度", width=80, anchor="center")
        self.result_table.column("坐标位置", width=180, anchor="center")

        table_scroll = Scrollbar(right_frame, orient="vertical", command=self.result_table.yview)
        table_scroll.pack(side="right", fill="y")
        self.result_table.configure(yscrollcommand=table_scroll.set)
        self.result_table.pack(fill="both", expand=True, padx=10, pady=5)

    # ======================== 新增：导出表格到项目目录 ========================
    def _export_table(self):
        # 1. 检查是否有数据
        if not self.result_table.get_children():
            messagebox.showwarning("无数据", "表格中暂无检测结果，无法导出")
            return

        # 2. 定义表格保存路径（项目目录下的plate_detection_result.csv）
        save_file = os.path.join(TABLE_SAVE_PATH, "plate_detection_result.csv")

        # 3. 写入CSV文件
        try:
            with open(save_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow([self.result_table.heading(col)["text"] for col in self.result_table["columns"]])
                # 写入每一行数据
                for item in self.result_table.get_children():
                    writer.writerow(self.result_table.item(item)["values"])
            messagebox.showinfo("导出成功", f"表格已保存到项目目录：\n{save_file}")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存表格时出错：\n{str(e)}")

    # ======================== 其余功能（无修改） ========================
    def _on_img_drop(self, event):
        if self.batch_running:
            messagebox.showwarning("批量处理中", "请先等待批量处理完成")
            return
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith((".jpg", ".png", ".jpeg")):
            self.img_path = file_path
            self._load_and_detect_img(file_path)
        else:
            messagebox.showwarning("无效文件", "请拖拽JPG/PNG格式的图片文件")

    def _select_single_img(self):
        if self.batch_running:
            messagebox.showwarning("批量处理中", "请先等待批量处理完成")
            return
        file_path = filedialog.askopenfilename(
            title="选择单张图片",
            filetypes=[("图像文件", "*.jpg;*.png;*.jpeg"), ("所有文件", "*.*")]
        )
        if file_path:
            self.img_path = file_path
            self._load_and_detect_img(file_path)

    def _load_and_detect_img(self, file_path):
        try:
            self.raw_img = cv2.imread(file_path)
            tk_img = self._img_to_tk(self.raw_img)
            self.img_label.config(image=tk_img, text="")
            self.img_label.image = tk_img
            threading.Thread(
                target=self._detect_thread,
                args=(file_path,),
                daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror("加载失败", f"图片加载错误：\n{str(e)}")

    def _select_folder(self):
        if self.batch_running:
            messagebox.showwarning("批量处理中", "请先等待批量处理完成")
            return
        folder_path = filedialog.askdirectory(title="选择批量识别文件夹")
        if not folder_path:
            return
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(img_files) == 0:
            messagebox.showwarning("无图片文件", "所选文件夹中没有JPG/PNG格式的图片")
            return
        self.batch_running = True
        self.pause_event.set()
        messagebox.showinfo("批量识别启动", f"共检测到 {len(img_files)} 张图片，开始批量识别...")
        threading.Thread(
            target=self._batch_detect_thread,
            args=(folder_path, img_files),
            daemon=True
        ).start()

    def _pause_detect(self):
        if not self.batch_running:
            messagebox.showwarning("未在批量检测", "请先启动批量检测后再操作")
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            messagebox.showinfo("暂停成功", "批量检测已暂停，点击“暂停检测”可恢复")
        else:
            self.pause_event.set()
            messagebox.showinfo("恢复成功", "批量检测已恢复")

    def _detect_thread(self, file_path):
        raw_img, result_img, plate_char, conf, coord = detect_recognize(
            file_path, self.yolo_model, self.plate_ocr, CONF_THRESH
        )
        self.result_img = result_img
        self.after(0, self._update_result, file_path, plate_char, conf, coord)

    def _batch_detect_thread(self, folder_path, img_files):
        for img_file in img_files:
            while not self.pause_event.is_set():
                time.sleep(0.1)
                if not self.batch_running:
                    break
            if not self.batch_running:
                break
            file_path = Path(folder_path) / img_file
            raw_img, result_img, plate_char, conf, coord = detect_recognize(
                str(file_path), self.yolo_model, self.plate_ocr, CONF_THRESH
            )
            if result_img is not None:
                name, ext = os.path.splitext(img_file)
                save_path = Path(OUTPUT_PATH) / f"{name}_batch_result{ext}"
                cv2.imwrite(str(save_path), result_img)
            self.after(0, self._update_table, str(file_path), plate_char, conf, coord)
            time.sleep(0.1)
        self.batch_running = False
        self.pause_event.set()
        self.after(0, lambda: messagebox.showinfo("批量识别完成",
                                                  f"共处理 {len(img_files)} 张图片，结果已保存至：\n{OUTPUT_PATH}"))

    def _update_result(self, file_path, plate_char, conf, coord):
        self.plate_char_var.set(plate_char)
        self.conf_var.set(f"{conf:.2f}")
        if coord:
            self.xmin_var.set(str(coord[0]))
            self.ymin_var.set(str(coord[1]))
            self.xmax_var.set(str(coord[2]))
            self.ymax_var.set(str(coord[3]))
        else:
            self.xmin_var.set("——")
            self.ymin_var.set("——")
            self.xmax_var.set("——")
            self.ymax_var.set("——")
        if self.result_img is not None:
            tk_img = self._img_to_tk(self.result_img)
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img
        self._update_table(file_path, plate_char, conf, coord)

    def _update_table(self, file_path, plate_char, conf, coord):
        self.detect_count += 1
        coord_str = f"[{coord[0]}, {coord[1]}, {coord[2]}, {coord[3]}]" if coord else "——"
        conf_str = f"{conf:.2f}" if conf > 0 else "0.00"
        self.result_table.insert("", "end", values=(
            self.detect_count,
            file_path,
            plate_char,
            conf_str,
            coord_str
        ))
        self.result_table.see("end")

    def _save_result(self):
        if self.result_img is None:
            messagebox.showwarning("无结果", "没有可保存的检测结果图片")
            return
        if not self.img_path:
            messagebox.showwarning("无图片", "请先选择并检测一张图片")
            return
        img_name = os.path.basename(self.img_path)
        name, ext = os.path.splitext(img_name)
        save_path = Path(OUTPUT_PATH) / f"{name}_result{ext}"
        cv2.imwrite(str(save_path), self.result_img)
        messagebox.showinfo("保存成功", f"检测结果已保存至：\n{save_path}")

    def _clear_table(self):
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        self.detect_count = 0
        messagebox.showinfo("清空成功", "检测结果表格已清空")

    def _img_to_tk(self, cv2_img):
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        max_width = self.img_display_frame.winfo_width() - 40
        max_height = self.img_display_frame.winfo_height() - 40
        if max_width > 0 and max_height > 0:
            pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)


# ======================== 主程序入口 ========================
if __name__ == "__main__":
    app = PlateRecognitionGUI()
    app.mainloop()