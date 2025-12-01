#!/usr/bin/env python3
"""
3D Gaussian Splatting Scene Reconstruction Application (Refactored)

Main GUI application for capturing RealSense data, processing with SfM (COLMAP),
and training 3D Gaussian Splatting models.
"""

import sys
import os
import datetime
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog,
    QSplitter, QToolBar, QAction, QDialog, QFormLayout,
    QSpinBox, QCheckBox, QMessageBox, QProgressDialog, QTextEdit,
    QDialogButtonBox, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon

import pyrealsense2 as rs

# Import modules
from gs_trainer import GsTrainer
from image_processor import ImageProcessor
from colmap_runner import ColmapRunner


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

    def initialize_camera(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            ctx = rs.context()
            if len(ctx.devices) == 0:
                self.error_occurred.emit("No RealSense device connected.")
                return False

            # Configure streams
            width, height = 640, 480
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.target_fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.target_fps)

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
        layout = QFormLayout()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.settings.get('target_fps', 30))
        layout.addRow("Target FPS:", self.fps_spin)
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(10, 5000)
        self.max_frames_spin.setValue(self.settings.get('max_frames', 200))
        layout.addRow("Max Frames (Processing):", self.max_frames_spin)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self.settings.get('viser_port', 7007))
        layout.addRow("Viser Port:", self.port_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)

    def get_settings(self):
        return {
            'target_fps': self.fps_spin.value(),
            'max_frames': self.max_frames_spin.value(),
            'min_frames': self.settings.get('min_frames', 50),
            'viser_port': self.port_spin.value()
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DGS Studio - SfM & Training")
        self.resize(1400, 800)

        # State
        self.output_dir = os.getcwd()
        self.current_recording_dir = None
        
        # Settings
        self.settings = {
            'max_frames': 200,
            'min_frames': 50,
            'target_fps': 30,
            'viser_port': 7007
        }

        # Workers
        self.realsense_worker = None
        self.gs_trainer = GsTrainer()
        self.colmap_runner = None

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
        
        # Process Images
        self.process_btn = QPushButton("1. Process Images (Blur Filter)")
        self.process_btn.clicked.connect(self.process_images)
        controls.addWidget(self.process_btn)
        
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
            self.settings = dialog.get_settings()
            # Update worker if running
            if self.realsense_worker:
                self.realsense_worker.target_fps = self.settings['target_fps']
                # Note: Changing FPS might require restarting pipeline, 
                # but we'll just set the var for next init or handle it if we want live update
                # For now, user might need to restart app or we add restart logic.
                # Let's try to restart camera if fps changed
                self.realsense_worker.stop()
                self.realsense_worker.wait()
                self.realsense_worker.start()
            
            self.statusBar().showMessage("Settings updated (Camera restarted)")

    def start_camera(self):
        self.realsense_worker = RealsenseWorker()
        self.realsense_worker.frame_ready.connect(self.on_frame_ready)
        self.realsense_worker.error_occurred.connect(lambda e: self.statusBar().showMessage(f"Camera Error: {e}"))
        self.realsense_worker.start()

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
        if not self.realsense_worker: return
        
        if checked:
            dir_path = self.realsense_worker.create_recording_dir(self.output_dir)
            self.current_recording_dir = dir_path
            self.realsense_worker.recording = True
            self.statusBar().showMessage(f"Recording to {dir_path}")
            self.out_label.setText(f"Current: {os.path.basename(dir_path)}")
        else:
            self.realsense_worker.recording = False
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

        self.sfm_dialog = ProcessingDialog("Generating Sparse PC (COLMAP)")
        self.sfm_dialog.show()
        
        self.colmap_runner = ColmapRunner(self.current_recording_dir)
        self.colmap_runner.progress_update.connect(self.sfm_dialog.log)
        self.colmap_runner.processing_finished.connect(self.on_sfm_finished)
        self.colmap_runner.start()

    def on_sfm_finished(self, success, msg):
        self.sfm_dialog.finish(success, msg)
        if success:
            self.load_sparse_pc()

    def load_sparse_pc(self):
        if not self.current_recording_dir: return
        
        # COLMAP output via Nerfstudio usually goes to 'sparse_pc.ply' or similar depending on version.
        # ns-process-data creates 'colmap/sparse/0/points3D.ply' sometimes, OR it just exports a .ply 
        # actually ns-process-data output structure:
        # data_dir/
        #   colmap/
        #   images/
        #   transforms.json
        #   sparse_pc.ply (sometimes)
        
        # We check for common locations
        candidates = [
            os.path.join(self.current_recording_dir, "sparse_pc.ply"),
            os.path.join(self.current_recording_dir, "colmap", "sparse", "0", "points3D.ply")
        ]
        
        ply_path = None
        for c in candidates:
            if os.path.exists(c):
                ply_path = c
                break
        
        if ply_path:
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                
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
                self.statusBar().showMessage(f"Loaded Sparse PC: {os.path.basename(ply_path)}")
            except Exception as e:
                print(f"Error loading PLY: {e}")
        else:
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
        
        outputs_dir = os.path.join(self.current_recording_dir, "outputs")
        if not os.path.exists(outputs_dir):
            QMessageBox.warning(self, "Error", "No outputs directory found. Did you train a model?")
            return

        # Find config.yml recursively
        config_files = []
        for root, dirs, files in os.walk(outputs_dir):
            if "config.yml" in files:
                config_files.append(os.path.join(root, "config.yml"))
        
        if not config_files:
            QMessageBox.warning(self, "Error", "No config.yml found in outputs.")
            return
            
        # Sort by modification time (newest first)
        config_files.sort(key=os.path.getmtime, reverse=True)
        config_path = config_files[0]
        
        export_path = os.path.join(self.current_recording_dir, "gs_export", "splat.ply")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        msg = f"Found config: {os.path.relpath(config_path, self.current_recording_dir)}\n\nExporting to: {export_path}"
        self.statusBar().showMessage("Exporting...")
        
        try:
            self.gs_trainer.export_ply(config_path, export_path, cwd=self.current_recording_dir)
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{export_path}")
            self.statusBar().showMessage(f"Exported to {os.path.basename(export_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def select_output_directory(self):
        if self.realsense_worker: self.realsense_worker.paused = True
        
        d = QFileDialog.getExistingDirectory(
            self, 
            "Select Output", 
            self.output_dir,
            options=QFileDialog.DontUseNativeDialog
        )
        
        if self.realsense_worker: self.realsense_worker.paused = False
        
        if d:
            self.output_dir = d
            self.out_label.setText(f"Path: {d}")

    def load_existing_capture(self):
        if self.realsense_worker: self.realsense_worker.paused = True

        d = QFileDialog.getExistingDirectory(
            self, 
            "Select Capture", 
            self.output_dir,
            options=QFileDialog.DontUseNativeDialog
        )
        
        if self.realsense_worker: self.realsense_worker.paused = False

        if d:
            self.current_recording_dir = d
            self.out_label.setText(f"Current: {os.path.basename(d)}")
            self.load_sparse_pc()
            self.statusBar().showMessage(f"Loaded {os.path.basename(d)}")

    def closeEvent(self, event):
        if self.realsense_worker:
            self.realsense_worker.stop()
        if self.gs_trainer.training:
            self.gs_trainer.stop()
        try:
            self.plotter.close()
        except:
            pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())