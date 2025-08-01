import os
import sys
import subprocess
import psutil
import yaml
import torch
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QProgressBar, QComboBox, QMessageBox, QRadioButton, QButtonGroup, QSlider, QCheckBox, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QPalette, QColor
from loguru import logger

CONFIG_FILE = "ui.yml"

class Worker(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, process):
        super().__init__()
        self.process = process

    def run(self):
        try:
            for line in iter(self.process.stdout.readline, ""):
                self.log_signal.emit(line)
                if "overall_progress:" in line:
                    progress = int(line.strip().split("overall_progress:")[1].strip())
                    self.progress_signal.emit(progress)

            self.process.stdout.close()
        finally:
            self.finished_signal.emit()


class RemwmWorker(QObject):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, remwm_main_func, args):
        super().__init__()
        self.remwm_main_func = remwm_main_func
        self.args = args
        self.thread = None
    
    def run(self):
        # Run remwm.main in a separate thread to avoid blocking the GUI
        self.thread = threading.Thread(target=self._run_remwm)
        self.thread.start()
    
    def _run_remwm(self):
        try:
            # Create a mock context object to capture print statements
            class MockContext:
                def __init__(self, log_signal):
                    self.log_signal = log_signal
                    
                def __enter__(self):
                    import sys
                    import io
                    self.old_stdout = sys.stdout
                    self.old_stderr = sys.stderr
                    self.output_buffer = io.StringIO()
                    self.error_buffer = io.StringIO()
                    sys.stdout = self.output_buffer
                    sys.stderr = self.error_buffer
                    return self
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    import sys
                    sys.stdout = self.old_stdout
                    sys.stderr = self.old_stderr
                    output = self.output_buffer.getvalue()
                    error_output = self.error_buffer.getvalue()
                    
                    # Emit stdout lines
                    for line in output.split('\n'):
                        if line.strip():
                            self.log_signal.emit(f"[STDOUT] {line}")
                    
                    # Emit stderr lines
                    for line in error_output.split('\n'):
                        if line.strip():
                            self.log_signal.emit(f"[STDERR] {line}")
                    
            # Run remwm.main with captured output
            with MockContext(self.log_signal):
                self.remwm_main_func(self.args)
        except Exception as e:
            self.log_signal.emit(f"Error running remwm: {str(e)}")
        finally:
            self.finished_signal.emit()

class WatermarkRemoverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水印去除工具")
        self.setGeometry(100, 100, 800, 600)

        # Initialize UI elements
        self.radio_single = QRadioButton("处理单张图片")
        self.radio_batch = QRadioButton("处理目录")
        self.radio_single.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_batch)

        self.input_path = QLineEdit(self)
        self.output_path = QLineEdit(self)
        self.overwrite_checkbox = QCheckBox("覆盖已存在的文件", self)
        self.overwrite_checkbox.setChecked(True)
        self.mosaic_checkbox = QCheckBox("应用马赛克效果", self)
        self.mosaic_checkbox.setChecked(True)
        self.mosaic_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.mosaic_size_slider.setRange(5, 50)
        self.mosaic_size_slider.setValue(15)
        self.mosaic_size_label = QLabel("马赛克大小: 15px", self)
        self.mosaic_size_slider.valueChanged.connect(lambda v: self.mosaic_size_label.setText(f"马赛克大小: {v}px"))
        
        self.max_bbox_percent_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.max_bbox_percent_slider.setRange(1, 100)
        self.max_bbox_percent_slider.setValue(10)
        self.max_bbox_percent_label = QLabel(f"最大边界框百分比: 10%", self)
        self.max_bbox_percent_slider.valueChanged.connect(self.update_bbox_label)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.sensitivity_slider.setRange(5, 20)  # 0.5–2.0 step 0.1
        self.sensitivity_slider.setValue(17)
        self.sensitivity_label = QLabel("敏感度: 1.7", self)
        self.sensitivity_slider.valueChanged.connect(lambda v: self.sensitivity_label.setText(f"敏感度: {v/10:.1f}"))

        self.rotated_checkbox = QCheckBox("包含旋转检测", self)
        self.rotated_checkbox.setChecked(True)  # 默认勾选旋转检测

        self.force_format_png = QRadioButton("PNG")
        self.force_format_webp = QRadioButton("WEBP")
        self.force_format_jpg = QRadioButton("JPG")
        self.force_format_none = QRadioButton("无")
        self.force_format_none.setChecked(True)
        self.force_format_group = QButtonGroup()
        self.force_format_group.addButton(self.force_format_png)
        self.force_format_group.addButton(self.force_format_webp)
        self.force_format_group.addButton(self.force_format_jpg)
        self.force_format_group.addButton(self.force_format_none)

        self.progress_bar = QProgressBar(self)
        self.logs = QTextEdit(self)
        self.logs.setReadOnly(True)
        self.logs.setVisible(False)

        self.start_button = QPushButton("开始", self)
        self.stop_button = QPushButton("停止", self)
        self.toggle_logs_button = QPushButton("显示日志", self)
        self.toggle_logs_button.setCheckable(True)
        self.stop_button.setDisabled(True)

        # Status bar for system info
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(1000)  # Update every second

        self.process = None
        self.thread = None

        # Layout
        layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_batch)

        # Input and output paths
        path_layout = QVBoxLayout()
        path_layout.addWidget(QLabel("输入路径:"))
        path_layout.addWidget(self.input_path)
        path_layout.addWidget(QPushButton("浏览", clicked=self.browse_input))
        path_layout.addWidget(QLabel("输出路径:"))
        path_layout.addWidget(self.output_path)
        path_layout.addWidget(QPushButton("浏览", clicked=self.browse_output))

        # Options
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.overwrite_checkbox)
        options_layout.addWidget(self.mosaic_checkbox)

        mosaic_layout = QVBoxLayout()
        mosaic_layout.addWidget(self.mosaic_size_label)
        mosaic_layout.addWidget(self.mosaic_size_slider)
        options_layout.addLayout(mosaic_layout)

        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.max_bbox_percent_label)
        bbox_layout.addWidget(self.max_bbox_percent_slider)
        options_layout.addLayout(bbox_layout)

        sens_layout = QVBoxLayout()
        sens_layout.addWidget(self.sensitivity_label)
        sens_layout.addWidget(self.sensitivity_slider)
        options_layout.addLayout(sens_layout)

        options_layout.addWidget(self.rotated_checkbox)

        force_format_layout = QHBoxLayout()
        force_format_layout.addWidget(QLabel("强制格式:"))
        force_format_layout.addWidget(self.force_format_png)
        force_format_layout.addWidget(self.force_format_webp)
        force_format_layout.addWidget(self.force_format_jpg)
        force_format_layout.addWidget(self.force_format_none)
        options_layout.addLayout(force_format_layout)

        # Logs and progress
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("进度:"))
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.toggle_logs_button)
        progress_layout.addWidget(self.logs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Final assembly
        layout.addLayout(mode_layout)
        layout.addLayout(path_layout)
        layout.addLayout(options_layout)
        layout.addLayout(progress_layout)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect buttons
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.toggle_logs_button.toggled.connect(self.toggle_logs)

        self.apply_dark_mode_if_needed()

        # Load configuration
        self.load_config()

    def update_bbox_label(self, value):
        self.max_bbox_percent_label.setText(f"最大边界框百分比: {value}%")

    def toggle_logs(self, checked):
        self.logs.setVisible(checked)
        self.toggle_logs_button.setText("隐藏日志" if checked else "显示日志")

    def apply_dark_mode_if_needed(self):
        if QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

            QApplication.instance().setPalette(dark_palette)

    def update_system_info(self):
        cuda_available = "CUDA: 可用" if torch.cuda.is_available() else "CUDA: 不可用"
        ram = psutil.virtual_memory()
        ram_usage = ram.used // (1024 ** 2)
        ram_total = ram.total // (1024 ** 2)
        ram_percentage = ram.percent

        vram_status = "不可用"
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            vram_total = gpu_info.total_memory // (1024 ** 2)
            vram_used = vram_total - (torch.cuda.memory_reserved(0) // (1024 ** 2))
            vram_percentage = (vram_used / vram_total) * 100
            vram_status = f"显存: {vram_used} MB / {vram_total} MB ({vram_percentage:.2f}%)"

        status_text = (
            f"{cuda_available} | 内存: {ram_usage} MB / {ram_total} MB ({ram_percentage}%) | {vram_status} | CPU负载: {psutil.cpu_percent()}%"
        )
        self.status_bar.showMessage(status_text)

    def browse_input(self):
        if self.radio_single.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择输入图片", "", "图片 (*.png *.jpg *.jpeg *.webp)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.input_path.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_path.setText(path)

    def start_processing(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()

        if not input_path or not output_path:
            QMessageBox.critical(self, "错误", "输入和输出路径是必需的。")
            return

        # Instead of running subprocess, we'll import and run remwm.py directly
        # Get the absolute path to remwm.py, handling PyInstaller bundling
        logger.info(f"当前工作目录: {os.getcwd()}")
        self.update_logs(f"当前工作目录: {os.getcwd()}")
        
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            script_dir = sys._MEIPASS
            logger.info(f"作为编译后的可执行文件运行, _MEIPASS: {script_dir}")
            self.update_logs(f"作为编译后的可执行文件运行, _MEIPASS: {script_dir}")
        else:
            # Running as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logger.info(f"作为脚本运行, script_dir: {script_dir}")
            self.update_logs(f"作为脚本运行, script_dir: {script_dir}")
        
        remwm_path = os.path.join(script_dir, "remwm.py")
        logger.info(f"remwm.py 路径: {remwm_path}")
        self.update_logs(f"remwm.py 路径: {remwm_path}")
        
        # 检查remwm.py文件是否存在
        if not os.path.exists(remwm_path):
            logger.error(f"remwm.py 未找到于: {remwm_path}")
            self.update_logs(f"remwm.py 未找到于: {remwm_path}")
            QMessageBox.critical(self, "错误", f"remwm.py 未找到于: {remwm_path}")
            return
        
        # Add the script directory to sys.path to allow importing remwm
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Import remwm and run it in a separate thread
        try:
            import remwm
            
            # Prepare arguments for remwm.main
            args = [input_path, output_path]
            if self.overwrite_checkbox.isChecked():
                args.append("--overwrite")
            if self.mosaic_checkbox.isChecked():
                args.append("--mosaic")
                args.append(f"--mosaic-size={self.mosaic_size_slider.value()}")
            args.append(f"--max-bbox-percent={self.max_bbox_percent_slider.value()}")
            args.append(f"--detection-sensitivity={self.sensitivity_slider.value()/10:.1f}")
            
            # Handle force format option
            force_format = "None"
            if self.force_format_png.isChecked():
                force_format = "PNG"
            elif self.force_format_webp.isChecked():
                force_format = "WEBP"
            elif self.force_format_jpg.isChecked():
                force_format = "JPG"
            
            if force_format != "None":
                args.append(f"--force-format={force_format}")
            
            if self.rotated_checkbox.isChecked():
                args.append("--include-rotated")
            
            # Log the arguments for debugging
            logger.info(f"运行 remwm.main 参数: {args}")
            self.update_logs(f"运行 remwm.main 参数: {args}")
            
            # Run remwm.main in a separate thread to avoid blocking the GUI
            self.worker_thread = QThread()
            self.worker = RemwmWorker(remwm.main, args)
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(lambda: self.worker.run())
            self.worker.log_signal.connect(self.update_logs)
            self.worker.finished_signal.connect(self.reset_ui)
            self.worker.finished_signal.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            
            self.worker_thread.start()
            
            self.stop_button.setDisabled(False)
            self.start_button.setDisabled(True)
        except Exception as e:
            logger.error(f"导入或运行 remwm 时出错: {e}")
            self.update_logs(f"导入或运行 remwm 时出错: {e}")
            QMessageBox.critical(self, "错误", f"运行 remwm 失败: {e}")

    def update_logs(self, line):
        self.logs.append(line.strip())
        self.logs.repaint()
        QApplication.processEvents()

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def stop_processing(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.reset_ui()

    def reset_ui(self):
        self.stop_button.setDisabled(True)
        self.start_button.setDisabled(False)
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.process = None
        self.thread = None

    def save_config(self):
        config = {
            "input_path": self.input_path.text(),
            "output_path": self.output_path.text(),
            "overwrite": self.overwrite_checkbox.isChecked(),
            "mosaic": self.mosaic_checkbox.isChecked(),
            "mosaic_size": self.mosaic_size_slider.value(),
            "max_bbox_percent": self.max_bbox_percent_slider.value(),
            "force_format": "PNG" if self.force_format_png.isChecked() else "WEBP" if self.force_format_webp.isChecked() else "JPG" if self.force_format_jpg.isChecked() else "None",
            "mode": "single" if self.radio_single.isChecked() else "batch"
        }
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
                self.input_path.setText(config.get("input_path", ""))
                self.output_path.setText(config.get("output_path", ""))
                self.overwrite_checkbox.setChecked(config.get("overwrite", False))
                self.mosaic_checkbox.setChecked(config.get("mosaic", False))
                self.mosaic_size_slider.setValue(config.get("mosaic_size", 20))
                self.max_bbox_percent_slider.setValue(config.get("max_bbox_percent", 10))
                force_format = config.get("force_format", "None")
                if force_format == "PNG":
                    self.force_format_png.setChecked(True)
                elif force_format == "WEBP":
                    self.force_format_webp.setChecked(True)
                elif force_format == "JPG":
                    self.force_format_jpg.setChecked(True)
                else:
                    self.force_format_none.setChecked(True)
                mode = config.get("mode", "single")
                if mode == "single":
                    self.radio_single.setChecked(True)
                else:
                    self.radio_batch.setChecked(True)

    def closeEvent(self, event):
        self.save_config()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = WatermarkRemoverGUI()
    gui.show()
    sys.exit(app.exec())