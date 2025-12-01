import subprocess
import os
import shutil
from PyQt5.QtCore import QThread, pyqtSignal

class ColmapRunner(QThread):
    """
    Runs COLMAP manually to avoid flag incompatibilities with Nerfstudio,
    then runs ns-process-data to package the results.
    """
    processing_finished = pyqtSignal(bool, str)
    progress_update = pyqtSignal(str)

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.process = None

    def run_cmd(self, cmd, description):
        """Helper to run a command and stream output"""
        self.progress_update.emit(f"\n--- {description} ---")
        self.progress_update.emit(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.progress_update.emit(line)
            
            process.wait()
            return process.returncode == 0
            
        except Exception as e:
            self.progress_update.emit(f"Error executing {description}: {e}")
            return False

    def run(self):
        images_dir = os.path.join(self.data_dir, "images")
        colmap_dir = os.path.join(self.data_dir, "colmap")
        db_path = os.path.join(colmap_dir, "database.db")
        sparse_dir = os.path.join(colmap_dir, "sparse")
        
        if not os.path.exists(images_dir):
            self.processing_finished.emit(False, "No images directory found.")
            return

        # Ensure clean directories
        os.makedirs(colmap_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)

        # 1. Feature Extractor
        # Note: Using --FeatureExtraction.use_gpu 1 (Correct for COLMAP 3.13+)
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
            "--FeatureExtraction.use_gpu", "1" 
        ]
        if not self.run_cmd(cmd_extract, "Extracting Features"):
            self.processing_finished.emit(False, "Feature extraction failed.")
            return

        # 2. Matcher
        # using sequential_matcher (loop detection disabled as we lack vocab tree)
        cmd_match = [
            "colmap", "sequential_matcher",
            "--database_path", db_path,
            "--FeatureMatching.use_gpu", "1",
            "--SequentialMatching.loop_detection", "0"
        ]
        if not self.run_cmd(cmd_match, "Matching Features"):
            self.processing_finished.emit(False, "Feature matching failed.")
            return

        # 3. Mapper (SfM)
        cmd_mapper = [
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir
        ]
        if not self.run_cmd(cmd_mapper, "Running SfM (Mapper)нка"):
            self.processing_finished.emit(False, "SfM Mapper failed.")
            return

        # 4. Verify Output
        # COLMAP mapper usually outputs to sparse/0 if successful
        reconstruction_path = os.path.join(sparse_dir, "0")
        if not os.path.exists(reconstruction_path):
             # sometimes it outputs directly to sparse if only one model
             if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
                 reconstruction_path = sparse_dir
             else:
                 self.processing_finished.emit(False, "SfM did not produce a reconstruction.")
                 return

        # 5. Nerfstudio Processing (Packaging)
        # We use --skip-colmap to just generate transforms.json
        cmd_ns = [
            "ns-process-data", "images",
            "--data", images_dir,
            "--output-dir", self.data_dir,
            "--skip-colmap",
            "--skip-image-processing", 
            "--colmap-model-path", reconstruction_path,
            "--verbose"
        ]
        if not self.run_cmd(cmd_ns, "Packaging for Nerfstudio"):
            self.processing_finished.emit(False, "Nerfstudio packaging failed.")
            return

        self.processing_finished.emit(True, "SfM Pipeline Complete!")

    def stop(self):
        # Since we use Popen inside run_cmd in a blocking way, 
        # true termination is harder without class member tracking.
        # But this is a basic implementation.
        pass