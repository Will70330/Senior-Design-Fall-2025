#!/usr/bin/env python3
"""
3D Gaussian Splatting Scene Reconstruction Application (Refactored)

Main GUI application for capturing RealSense data, processing with SfM (COLMAP),
and training 3D Gaussian Splatting models.
"""

import sys
import os

# Add venv site-packages to path to allow importing modules (Open3D, etc.) 
# when running with system python (for ROS 2)
current_dir = os.path.dirname(os.path.abspath(__file__))
virtualenv_site_packages = os.path.join(
    current_dir, 
    'venv', 
    'lib', 
    f'python{sys.version_info.major}.{sys.version_info.minor}', 
    'site-packages'
)
if os.path.exists(virtualenv_site_packages):
    sys.path.insert(0, virtualenv_site_packages)

import datetime
import json
import numpy as np
import cv2
import open3d as o3d
import struct
from pathlib import Path
import time

# ROS 2 Imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    import message_filters
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS 2 (rclpy) not found. Wireless mode will be disabled.")

import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog,
    QSplitter, QToolBar, QAction, QDialog, QFormLayout,
    QSpinBox, QCheckBox, QMessageBox, QProgressDialog, QTextEdit,
    QDialogButtonBox, QProgressBar, QGroupBox, QGridLayout, QTabWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon

import pyrealsense2 as rs

# Import modules
from gs_trainer import GsTrainer
from image_processor import ImageProcessor
from colmap_runner import ColmapRunner
from metrics_calculator import MetricsCalculator


class RosRealsenseWorker(QThread):
    """Background thread for Wireless RealSense capture via ROS 2"""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict)  # color, depth, stats
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        self.recording = False
        self.node = None
        
        # Recording state
        self.record_dir = None
        self.images_dir = None
        self.frame_idx = 0

        # Settings (updated by main thread)
        self.target_fps = 30
        self.width = 640
        self.height = 480

    def create_recording_dir(self, base_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = os.path.join(base_dir, f"capture_{timestamp}")
        self.images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.frame_idx = 0
        return self.record_dir

    def callback(self, color_msg, depth_msg):
        if self.paused:
            return

        try:
            # Decode Color (BGR8)
            color_arr = np.frombuffer(color_msg.data, dtype=np.uint8)
            color_image = color_arr.reshape((color_msg.height, color_msg.width, 3))
            
            # Decode Depth (16UC1)
            depth_arr = np.frombuffer(depth_msg.data, dtype=np.uint16)
            depth_image = depth_arr.reshape((depth_msg.height, depth_msg.width))

            # Recording logic
            if self.recording and self.images_dir:
                cv2.imwrite(os.path.join(self.images_dir, f"frame_{self.frame_idx:05d}.jpg"), color_image)
                cv2.imwrite(os.path.join(self.images_dir, f"depth_{self.frame_idx:05d}.png"), depth_image)
                self.frame_idx += 1

            stats = {
                'frame_idx': self.frame_idx if self.recording else 0,
                'recording': self.recording,
            }
            
            self.frame_ready.emit(color_image, depth_image, stats)
            
            # Determine connection status by data flow
            self.connection_status.emit(True)
            
        except Exception as e:
            print(f"Frame decode error: {e}")

    def run(self):
        if not ROS_AVAILABLE:
            self.error_occurred.emit("ROS 2 libraries not found.")
            return

        try:
            if not rclpy.ok():
                rclpy.init()
        except Exception as e:
            print(f"RCLPY Init error: {e}")

        self.node = rclpy.create_node('laptop_visualizer')
        
        # Subscribers
        color_sub = message_filters.Subscriber(self.node, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self.node, Image, '/camera/depth/image_raw')
        
        # Sync (Approximate because timestamps might drift slightly over wireless)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.callback)
        
        self.running = True
        self.connection_status.emit(True) # Optimistic init

        while self.running and rclpy.ok():
            # Spin once to process callbacks
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        self.node.destroy_node()
        # Do not rclpy.shutdown() here if other nodes might exist, 
        # but for this app it's likely fine or handled by global cleanup.

    def stop(self):
        self.running = False
        self.wait()


class RealsenseWorker(QThread):
    """Background thread for RealSense capture only (no odometry)"""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict)  # color, depth, stats
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        self.recording = False

        self.pipeline = None
        self.config = None
        self.align = None
        
        # Recording state
        self.record_dir = None
        self.images_dir = None
        self.frame_idx = 0

        # Settings
        self.target_fps = 30
        self.width = 640
        self.height = 480

    def initialize_camera(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            ctx = rs.context()
            if len(ctx.devices) == 0:
                self.error_occurred.emit("No RealSense device connected.")
                return False

            # Configure streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.target_fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.target_fps)

            profile = self.pipeline.start(self.config)
            
            # Configure for motion (Global shutter emulation/Auto-exposure priority)
            try:
                device = profile.get_device()
                depth_sensor = device.first_depth_sensor()
                color_sensor = device.first_color_sensor()
                
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                if color_sensor.supports(rs.option.auto_exposure_priority):
                    color_sensor.set_option(rs.option.auto_exposure_priority, 0.0)
            except Exception as e:
                print(f"Warning: Could not configure camera options: {e}")

            self.align = rs.align(rs.stream.color)
            return True

        except Exception as e:
            self.error_occurred.emit(f"Camera init failed: {str(e)}")
            return False

    def create_recording_dir(self, base_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = os.path.join(base_dir, f"capture_{timestamp}")
        self.images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.frame_idx = 0
        return self.record_dir

    def run(self):
        if not self.initialize_camera():
            return

        self.running = True
        frame_count = 0

        while self.running:
            if self.paused:
                self.msleep(100)
                continue

            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Recording logic
                if self.recording and self.images_dir:
                    cv2.imwrite(os.path.join(self.images_dir, f"frame_{self.frame_idx:05d}.jpg"), color_image)
                    cv2.imwrite(os.path.join(self.images_dir, f"depth_{self.frame_idx:05d}.png"), depth_image)
                    self.frame_idx += 1

                stats = {
                    'frame_idx': self.frame_idx if self.recording else frame_count,
                    'recording': self.recording,
                }
                
                self.frame_ready.emit(color_image, depth_image, stats)
                frame_count += 1

            except Exception as e:
                self.error_occurred.emit(f"Capture error: {str(e)}")
                break

        if self.pipeline:
            self.pipeline.stop()

    def stop(self):
        self.running = False


class ProcessingDialog(QDialog):
    """Dialog to show log output from background tasks (SfM, Processing)"""
    def __init__(self, title="Processing"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indeterminate
        layout.addWidget(self.progress)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        layout.addWidget(self.close_btn)
        
        self.setLayout(layout)

    def log(self, msg):
        self.log_text.append(msg)
        QApplication.processEvents()

    def finish(self, success, msg):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.log(f"\nResult: {msg}")
        self.close_btn.setEnabled(True)


class SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setModal(True)
        
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # --- Tab 1: Camera & Connection ---
        self.cam_tab = QWidget()
        cam_layout = QFormLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Local USB", "Wireless (ROS 2)"])
        self.mode_combo.setCurrentText(self.settings.get('mode', "Local USB"))
        cam_layout.addRow("Connection Mode:", self.mode_combo)
        
        # ROS 2 uses discovery, IP optional but kept for UI consistency or future advanced config
        self.ip_input = QLineEdit()
        self.ip_input.setText(self.settings.get('ip_address', ""))
        self.ip_input.setPlaceholderText("Auto-Discovery (Leave Empty)")
        self.ip_input.setEnabled(False) # Disabled for ROS 2 auto-discovery
        cam_layout.addRow("Camera IP:", self.ip_input)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.settings.get('target_fps', 30))
        cam_layout.addRow("Target FPS:", self.fps_spin)
        
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "848x480", "1280x720", "424x240"])
        cur_res = f"{self.settings.get('width', 640)}x{self.settings.get('height', 480)}"
        self.res_combo.setCurrentText(cur_res)
        cam_layout.addRow("Resolution:", self.res_combo)

        self.cam_tab.setLayout(cam_layout)
        self.tabs.addTab(self.cam_tab, "Camera")

        # --- Tab 2: Processing ---
        self.proc_tab = QWidget()
        proc_layout = QFormLayout()
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(10, 5000)
        self.max_frames_spin.setValue(self.settings.get('max_frames', 200))
        proc_layout.addRow("Max Frames (Processing):", self.max_frames_spin)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self.settings.get('viser_port', 7007))
        proc_layout.addRow("Viser Port:", self.port_spin)
        
        self.coord_check = QCheckBox()
        self.coord_check.setChecked(self.settings.get('adjust_coord', False))
        proc_layout.addRow("Adjust Coordinate Frame:", self.coord_check)

        self.proc_tab.setLayout(proc_layout)
        self.tabs.addTab(self.proc_tab, "Processing")

        self.layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        self.setLayout(self.layout)
        
        # Trigger mode update
        self.mode_combo.currentTextChanged.connect(self.toggle_ip_field)
        self.toggle_ip_field(self.mode_combo.currentText())

    def toggle_ip_field(self, text):
        pass # Kept for compatibility, but currently no-op as we force auto-discovery

    def get_settings(self):
        w, h = map(int, self.res_combo.currentText().split('x'))
        return {
            'mode': self.mode_combo.currentText(),
            'ip_address': self.ip_input.text(),
            'target_fps': self.fps_spin.value(),
            'width': w,
            'height': h,
            'max_frames': self.max_frames_spin.value(),
            'min_frames': self.settings.get('min_frames', 50),
            'viser_port': self.port_spin.value(),
            'adjust_coord': self.coord_check.isChecked()
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DGS Studio - SfM & Training (ROS 2 Enabled)")
        self.resize(1600, 1000)

        # State
        self.output_dir = os.getcwd()
        self.current_recording_dir = None
        
        # Settings
        self.settings = {
            'mode': 'Local USB',
            'ip_address': '',
            'max_frames': 200,
            'min_frames': 50,
            'target_fps': 30,
            'width': 640,
            'height': 480,
            'viser_port': 7007,
            'adjust_coord': False
        }

        # Workers
        self.camera_worker = None 
        self.gs_trainer = GsTrainer()
        self.colmap_runner = None
        self.metrics_calculator = MetricsCalculator()

        self.init_ui()
        self.start_camera()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Top Toolbar ---
        top_bar = QHBoxLayout()
        
        self.out_btn = QPushButton("Output Dir")
        self.out_btn.clicked.connect(self.select_output_directory)
        top_bar.addWidget(self.out_btn)
        
        self.load_btn = QPushButton("📂 Load Capture")
        self.load_btn.clicked.connect(self.load_existing_capture)
        top_bar.addWidget(self.load_btn)
        
        self.out_label = QLabel(f"Path: {self.output_dir}")
        top_bar.addWidget(self.out_label)
        
        top_bar.addStretch() 
        
        self.conn_status_label = QLabel("Initializing...")
        self.conn_status_label.setStyleSheet("font-weight: bold; color: gray;")
        top_bar.addWidget(self.conn_status_label)
        
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        top_bar.addWidget(self.settings_btn)

        main_layout.addLayout(top_bar)

        # --- Viewers ---
        splitter = QSplitter(Qt.Horizontal)
        
        # Camera Feed
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #555;")
        splitter.addWidget(self.camera_label)
        
        # 3D Viewer (PyVista)
        self.plotter = QtInteractor(splitter)
        self.plotter.setMinimumSize(400, 300)
        self.plotter.set_background('black')
        self.plotter.add_axes()
        splitter.addWidget(self.plotter.interactor)
        self.pv_actor = None
        
        main_layout.addWidget(splitter, 1)

        # --- Metrics Panel ---
        self.metrics_group = QGroupBox("Pipeline Metrics")
        self.metrics_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
        """)

        metrics_main_layout = QVBoxLayout()
        metrics_main_layout.setSpacing(15)
        metrics_main_layout.setContentsMargins(20, 20, 20, 15)

        # Metrics values in a horizontal layout
        metrics_values_layout = QHBoxLayout()
        metrics_values_layout.setSpacing(40)

        self.metric_labels = {}
        metric_items = [
            ('num_matched_frames', 'Matched Frames'),
            ('num_sparse_points', 'Sparse Points'),
            ('num_gaussians', 'Gaussians'),
            ('psnr', 'PSNR'),
            ('ssim', 'SSIM'),
        ]

        for key, label_text in metric_items:
            # Create a vertical layout for each metric
            metric_box = QVBoxLayout()
            metric_box.setSpacing(5)

            # Label
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 15px; font-weight: bold; color: #000000;")
            label.setAlignment(Qt.AlignCenter)
            metric_box.addWidget(label)

            # Value
            value_label = QLabel("--")
            value_label.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                color: #4CAF50;
                padding: 8px 15px;
                background-color: #2a2a2a;
                border-radius: 5px;
                min-width: 100px;
            """)
            value_label.setAlignment(Qt.AlignCenter)
            metric_box.addWidget(value_label)
            self.metric_labels[key] = value_label

            metrics_values_layout.addLayout(metric_box)

        metrics_main_layout.addLayout(metrics_values_layout)

        # Metrics buttons
        metrics_btn_layout = QHBoxLayout()
        metrics_btn_layout.setSpacing(15)

        self.calc_metrics_btn = QPushButton("Calculate Metrics")
        self.calc_metrics_btn.clicked.connect(self.calculate_metrics)
        self.calc_metrics_btn.setToolTip("Calculate all available metrics (fast, skips PSNR/SSIM)")
        self.calc_metrics_btn.setStyleSheet("padding: 8px 16px;")
        metrics_btn_layout.addWidget(self.calc_metrics_btn)

        self.calc_psnr_btn = QPushButton("Calculate PSNR/SSIM")
        self.calc_psnr_btn.clicked.connect(self.calculate_psnr_ssim)
        self.calc_psnr_btn.setToolTip("Calculate PSNR and SSIM (requires trained model, slower)")
        self.calc_psnr_btn.setStyleSheet("padding: 8px 16px;")
        metrics_btn_layout.addWidget(self.calc_psnr_btn)

        self.export_metrics_btn = QPushButton("Export CSV")
        self.export_metrics_btn.clicked.connect(self.export_metrics)
        self.export_metrics_btn.setToolTip("Export metrics to gs_export/metrics.csv")
        self.export_metrics_btn.setStyleSheet("padding: 8px 16px;")
        metrics_btn_layout.addWidget(self.export_metrics_btn)

        metrics_btn_layout.addStretch()
        metrics_main_layout.addLayout(metrics_btn_layout)

        self.metrics_group.setLayout(metrics_main_layout)
        self.metrics_group.setMinimumHeight(160)
        main_layout.addWidget(self.metrics_group)

        # --- Bottom Controls ---
        controls = QHBoxLayout()
        
        # Record
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("padding: 10px; font-weight: bold;")
        controls.addWidget(self.record_btn)
        
        controls.addStretch() # Separator
        
        # Reset Images
        self.reset_imgs_btn = QPushButton("↺ Reset Images")
        self.reset_imgs_btn.clicked.connect(self.reset_images)
        controls.addWidget(self.reset_imgs_btn)

        # Preprocess Image Names
        self.rename_imgs_btn = QPushButton("Convert Names")
        self.rename_imgs_btn.clicked.connect(self.rename_images_for_training)
        self.rename_imgs_btn.setToolTip("Renames 'image_XXXX.jpg/png' to 'frame_YYYYY.jpg/png' and re-indexes for training.")
        controls.addWidget(self.rename_imgs_btn)
        
        # Process Images
        self.process_btn = QPushButton("1. Process Images (Blur Filter)")
        self.process_btn.clicked.connect(self.process_images)
        controls.addWidget(self.process_btn)
        
        # Matcher Type
        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(["Sequential", "Exhaustive"])
        self.matcher_combo.setToolTip("Feature matching strategy")
        controls.addWidget(self.matcher_combo)

        # Generate PC
        self.sfm_btn = QPushButton("2. Generate Sparse PC (SfM)")
        self.sfm_btn.clicked.connect(self.run_sfm)
        controls.addWidget(self.sfm_btn)
        
        # Train
        self.train_btn = QPushButton("3. Train 3DGS")
        self.train_btn.clicked.connect(self.run_training)
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        controls.addWidget(self.train_btn)
        
        # Export
        self.export_btn = QPushButton("4. Export PLY")
        self.export_btn.clicked.connect(self.run_export)
        controls.addWidget(self.export_btn)
        
        main_layout.addLayout(controls)
        self.statusBar().showMessage("Ready")

    def show_settings(self):
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            old_mode = self.settings.get('mode')
            self.settings = new_settings
            
            # Restart camera if settings changed that affect it
            if self.camera_worker:
                self.camera_worker.stop()
                self.camera_worker.wait()
            
            self.start_camera()
            
            self.statusBar().showMessage("Settings updated")

    def rename_images_for_training(self):
        if not self.current_recording_dir:
            QMessageBox.warning(self, "Error", "No recording selected. Please load or record a capture first.")
            return

        image_dir = os.path.join(self.current_recording_dir, "images")
        if not os.path.exists(image_dir):
            QMessageBox.warning(self, "Error", "Images folder not found in the current capture directory.")
            return
        
        self.statusBar().showMessage("Renaming images... please wait.")
        QApplication.processEvents()

        processor = ImageProcessor()
        success, message = processor.rename_and_reindex_images(image_dir)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.statusBar().showMessage(f"Image renaming complete: {message}")
        else:
            QMessageBox.critical(self, "Error", message)
            self.statusBar().showMessage(f"Image renaming failed: {message}")

    def start_camera(self):
        mode = self.settings.get('mode')
        
        if "Wireless" in mode:
            if not ROS_AVAILABLE:
                QMessageBox.critical(self, "Error", "ROS 2 is not available in this environment.")
                return
            self.camera_worker = RosRealsenseWorker()
            self.conn_status_label.setText("Waiting for ROS2...")
            self.camera_worker.connection_status.connect(self.on_connection_status)
        else:
            self.camera_worker = RealsenseWorker()
            self.conn_status_label.setText("Local Camera")
            
        # Apply settings
        self.camera_worker.target_fps = self.settings.get('target_fps', 30)
        self.camera_worker.width = self.settings.get('width', 640)
        self.camera_worker.height = self.settings.get('height', 480)

        self.camera_worker.frame_ready.connect(self.on_frame_ready)
        self.camera_worker.error_occurred.connect(lambda e: self.statusBar().showMessage(f"Camera Error: {e}"))
        self.camera_worker.start()

    def on_connection_status(self, connected):
        if connected:
            self.conn_status_label.setText("ROS2: Receiving 🟢")
            self.conn_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.conn_status_label.setText("ROS2: Waiting 🔴")
            self.conn_status_label.setStyleSheet("color: red; font-weight: bold;")

    def on_frame_ready(self, color, depth, stats):
        # Update Image
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(pixmap)
        
        # Update Status
        rec_text = f"Recording: {stats['frame_idx']}" if stats['recording'] else "Live"
        color_hex = "#ff0000" if stats['recording'] else "#00ff00"
        self.record_btn.setStyleSheet(f"background-color: {color_hex}; color: white; padding: 10px; font-weight: bold;")
        if stats['recording']:
            self.record_btn.setText(f"Stop Recording ({stats['frame_idx']})")
        else:
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("padding: 10px;")

    def toggle_recording(self, checked):
        if not self.camera_worker: return
        
        if checked:
            dir_path = self.camera_worker.create_recording_dir(self.output_dir)
            self.current_recording_dir = dir_path
            self.camera_worker.recording = True
            self.statusBar().showMessage(f"Recording to {dir_path}")
            self.out_label.setText(f"Current: {os.path.basename(dir_path)}")
        else:
            self.camera_worker.recording = False
            self.statusBar().showMessage(f"Recording saved to {self.current_recording_dir}")

    def reset_images(self):
        if not self.current_recording_dir:
            return
            
        img_dir = os.path.join(self.current_recording_dir, "images")
        rejected_dir = os.path.join(self.current_recording_dir, "rejected_images")
        
        if not os.path.exists(rejected_dir):
            QMessageBox.information(self, "Info", "No rejected images found to restore.")
            return
            
        import shutil
        count = 0
        try:
            for f in os.listdir(rejected_dir):
                src = os.path.join(rejected_dir, f)
                dst = os.path.join(img_dir, f)
                if os.path.isfile(src):
                    shutil.move(src, dst)
                    count += 1
            os.rmdir(rejected_dir)
            QMessageBox.information(self, "Success", f"Restored {count} images.")
            self.statusBar().showMessage(f"Restored {count} images.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to restore images: {e}")

    def process_images(self):
        if not self.current_recording_dir:
            QMessageBox.warning(self, "Error", "No recording selected.")
            return
            
        img_dir = os.path.join(self.current_recording_dir, "images")
        if not os.path.exists(img_dir):
             QMessageBox.warning(self, "Error", "Images folder not found.")
             return
             
        processor = ImageProcessor()
        
        self.statusBar().showMessage("Processing images... please wait.")
        QApplication.processEvents()
        
        kept, rejected = processor.process_images(
            img_dir, 
            max_frames=self.settings['max_frames'],
            min_frames=self.settings['min_frames']
        )
        
        QMessageBox.information(self, "Result", f"Image Processing Complete.\nKept: {kept}\nRejected: {rejected}")
        self.statusBar().showMessage(f"Images processed. Kept {kept} frames.")

    def run_sfm(self):
        if not self.current_recording_dir:
            QMessageBox.warning(self, "Error", "No recording selected.")
            return

        matcher = self.matcher_combo.currentText().lower()
        self.sfm_dialog = ProcessingDialog(f"Generating Sparse PC (COLMAP - {matcher})")
        self.sfm_dialog.show()
        
        self.colmap_runner = ColmapRunner(self.current_recording_dir, matcher_type=matcher)
        self.colmap_runner.progress_update.connect(self.sfm_dialog.log)
        self.colmap_runner.processing_finished.connect(self.on_sfm_finished)
        self.colmap_runner.start()

    def on_sfm_finished(self, success, msg):
        self.sfm_dialog.finish(success, msg)
        if success:
            self.load_sparse_pc()
            # Auto-calculate SfM metrics
            self.metrics_calculator.set_data_dir(self.current_recording_dir)
            self.metrics_calculator.count_matched_frames()
            self.metrics_calculator.count_sparse_points()
            self.update_metrics_display()
            self.statusBar().showMessage("SfM complete. Metrics updated.")

    def load_sparse_pc(self):
        if not self.current_recording_dir: return
        
        # COLMAP output via Nerfstudio usually goes to 'sparse_pc.ply' or similar depending on version.
        # ns-process-data creates 'colmap/sparse/0/points3D.ply' sometimes, OR it just exports a .ply 
        # actually ns-process-data output structure:
        # Check for common sparse point cloud locations
        candidates = [
            os.path.join(self.current_recording_dir, "sparse_points.ply"),
            os.path.join(self.current_recording_dir, "sparse_pc.ply"),
            os.path.join(self.current_recording_dir, "colmap", "sparse", "0", "points3D.ply"),
            os.path.join(self.current_recording_dir, "sparse", "0", "points3D.ply"),
            os.path.join(self.current_recording_dir, "points3D.ply"),
        ]

        ply_path = None
        for c in candidates:
            if os.path.exists(c):
                ply_path = c
                print(f"Found sparse point cloud: {c}")
                break

        if ply_path:
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                if self.settings.get('adjust_coord', False):
                    # Transform [x, -z, -y] -> [x, y, z]
                    points = points[:, [0, 2, 1]]
                    points[:, 1] = -points[:, 1]
                    points[:, 2] = -points[:, 2]

                if self.pv_actor:
                    self.plotter.remove_actor(self.pv_actor)

                pv_cloud = pv.PolyData(points)
                if len(colors) > 0:
                    pv_cloud['colors'] = (colors * 255).astype(np.uint8)

                self.pv_actor = self.plotter.add_points(
                    pv_cloud,
                    scalars='colors',
                    rgb=True,
                    point_size=3
                )
                self.plotter.reset_camera()
                self.statusBar().showMessage(f"Loaded Sparse PC: {os.path.basename(ply_path)} ({len(points):,} points)")
            except Exception as e:
                print(f"Error loading PLY: {e}")
                self.statusBar().showMessage(f"Error loading PLY: {e}")
        else:
            print(f"Sparse PC not found. Checked: {candidates}")
            self.statusBar().showMessage("Sparse PC not found (check colmap output).")

    def run_training(self):
        if not self.current_recording_dir: return
        
        msg = f"Start training on: {os.path.basename(self.current_recording_dir)}?"
        if QMessageBox.question(self, "Train", msg) != QMessageBox.Yes:
            return
            
        # Check for transforms.json
        if not os.path.exists(os.path.join(self.current_recording_dir, "transforms.json")):
             QMessageBox.warning(self, "Error", "transforms.json not found. Did you run SfM?")
             return

        self.gs_trainer.train(self.current_recording_dir, use_masked=False)
        
        viewer_url = f"http://localhost:{self.settings['viser_port']}"
        QMessageBox.information(self, "Training", f"Training started!\nViewer: {viewer_url}")

    def run_export(self):
        if not self.current_recording_dir:
            QMessageBox.warning(self, "Error", "No recording selected.")
            return
            
        # 1. Find the config file
        outputs_dir = os.path.join(self.current_recording_dir, "outputs")
        if not os.path.exists(outputs_dir):
            QMessageBox.warning(self, "Error", "No trained model found (outputs dir missing).")
            return
            
        # Find all config.yml files recursively
        configs = []
        for root, dirs, files in os.walk(outputs_dir):
            if "config.yml" in files:
                configs.append(os.path.join(root, "config.yml"))
                
        if not configs:
            QMessageBox.warning(self, "Error", "No config.yml found in outputs.")
            return
            
        # Get the most recent config
        best_config = max(configs, key=os.path.getmtime)
        
        # 2. Define export path
        export_dir = os.path.join(self.current_recording_dir, "gs_export")
        os.makedirs(export_dir, exist_ok=True)
        # gs_trainer.export_ply uses dirname of the path passed to it
        dummy_path = os.path.join(export_dir, "point_cloud.ply")
        
        # 3. Run export
        self.statusBar().showMessage(f"Exporting from {os.path.basename(os.path.dirname(best_config))}...")
        QApplication.processEvents() # Update UI
        
        try:
            self.gs_trainer.export_ply(
                config_path=best_config,
                output_path=dummy_path,
                cwd=self.current_recording_dir
            )
            # Auto-calculate metrics to update Gaussian count
            self.calculate_metrics()
            
            QMessageBox.information(self, "Success", f"Export complete!\nSaved to: {export_dir}")
            self.statusBar().showMessage(f"Export saved to {export_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed: {e}")
            self.statusBar().showMessage("Export failed.")

    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.out_label.setText(f"Path: {self.output_dir}")

    def load_existing_capture(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Capture Directory", self.output_dir)
        if dir_path:
            self.current_recording_dir = dir_path
            self.out_label.setText(f"Current: {os.path.basename(dir_path)}")
            self.load_sparse_pc()
            
            # Reset metrics
            for k in self.metric_labels:
                self.metric_labels[k].setText("--")
                self.metric_labels[k].setStyleSheet("""
                    font-size: 18px;
                    font-weight: bold;
                    color: #4CAF50;
                    padding: 8px 15px;
                    background-color: #2a2a2a;
                    border-radius: 5px;
                    min-width: 100px;
                """)
            
            # Load existing metrics
            self.metrics_calculator.set_data_dir(self.current_recording_dir)
            self.calculate_metrics()

    def calculate_metrics(self):
        if not self.current_recording_dir:
            return
        
        self.metrics_calculator.set_data_dir(self.current_recording_dir)
        self.metrics_calculator.count_matched_frames()
        self.metrics_calculator.count_sparse_points()
        self.metrics_calculator.count_gaussians()
        self.update_metrics_display()

    def calculate_psnr_ssim(self):
        if not self.current_recording_dir:
            return
            
        self.statusBar().showMessage("Calculating PSNR/SSIM (this may take a while)...")
        QApplication.processEvents()

        self.metrics_calculator.set_data_dir(self.current_recording_dir)
        psnr, ssim = self.metrics_calculator.calculate_psnr_ssim()
        self.update_metrics_display()

        if psnr is not None and ssim is not None:
            QMessageBox.information(self, "Metrics Calculated", f"PSNR: {psnr:.2f}\nSSIM: {ssim:.4f}")
            self.statusBar().showMessage(f"Metrics: PSNR {psnr:.2f}, SSIM {ssim:.4f}")
        else:
            QMessageBox.warning(self, "Error", "Could not calculate PSNR/SSIM.\nCheck if model is trained (config.yml needed).")
            self.statusBar().showMessage("Metrics calculation failed.")

    def export_metrics(self):
        if not self.current_recording_dir:
            return
        self.metrics_calculator.set_data_dir(self.current_recording_dir)
        output_file = self.metrics_calculator.export_metrics()
        QMessageBox.information(self, "Export", f"Metrics exported to:\n{output_file}")

    def update_metrics_display(self):
        metrics = self.metrics_calculator.get_metrics()
        for key, value in metrics.items():
            if key == 'num_matched_frames' and metrics.get('total_input_frames') is not None:
                self.metric_labels[key].setText(f"{value} / {metrics['total_input_frames']}")
            elif key in self.metric_labels:
                if isinstance(value, (int, float)):
                    # Check for float values and format them to 2 decimal places
                    if isinstance(value, float):
                        self.metric_labels[key].setText(f"{value:.2f}")
                    else:
                        self.metric_labels[key].setText(str(value))
                else:
                    self.metric_labels[key].setText(str(value))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
