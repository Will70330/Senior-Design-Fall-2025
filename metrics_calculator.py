"""
Metrics Calculator for 3DGS Pipeline

Calculates and exports various metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Number of Gaussians in exported model
- Number of points in sparse point cloud
- Number of matched frames from COLMAP
"""

import os
import struct
import csv
import subprocess
import json
from datetime import datetime
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class MetricsCalculator:
    """Calculates and stores pipeline metrics"""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.metrics = {
            'psnr': None,
            'ssim': None,
            'num_gaussians': None,
            'num_sparse_points': None,
            'num_matched_frames': None,
            'total_input_frames': None,
            'timestamp': None
        }

    def set_data_dir(self, data_dir):
        """Set the data directory for metrics calculation"""
        self.data_dir = data_dir

    def read_colmap_images_binary(self, path):
        """
        Read COLMAP images.bin file to get matched/registered images.
        Returns dict of {image_id: (qvec, tvec, camera_id, name, points2D)}

        Based on COLMAP's binary format specification.
        """
        images = {}
        with open(path, "rb") as f:
            num_reg_images = struct.unpack('<Q', f.read(8))[0]

            for _ in range(num_reg_images):
                image_id = struct.unpack('<I', f.read(4))[0]
                qvec = struct.unpack('<4d', f.read(32))
                tvec = struct.unpack('<3d', f.read(24))
                camera_id = struct.unpack('<I', f.read(4))[0]

                # Read image name (null-terminated string)
                name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name += char
                name = name.decode('utf-8')

                # Read 2D points
                num_points2D = struct.unpack('<Q', f.read(8))[0]
                points2D = []
                for _ in range(num_points2D):
                    x, y = struct.unpack('<2d', f.read(16))
                    point3D_id = struct.unpack('<q', f.read(8))[0]
                    points2D.append((x, y, point3D_id))

                images[image_id] = {
                    'qvec': qvec,
                    'tvec': tvec,
                    'camera_id': camera_id,
                    'name': name,
                    'points2D': points2D
                }

        return images

    def read_colmap_points3D_binary(self, path):
        """
        Read COLMAP points3D.bin file.
        Returns dict of {point3D_id: (xyz, rgb, error, track)}
        """
        points3D = {}
        with open(path, "rb") as f:
            num_points = struct.unpack('<Q', f.read(8))[0]

            for _ in range(num_points):
                point3D_id = struct.unpack('<Q', f.read(8))[0]
                xyz = struct.unpack('<3d', f.read(24))
                rgb = struct.unpack('<3B', f.read(3))
                error = struct.unpack('<d', f.read(8))[0]

                track_length = struct.unpack('<Q', f.read(8))[0]
                track = []
                for _ in range(track_length):
                    image_id = struct.unpack('<I', f.read(4))[0]
                    point2D_idx = struct.unpack('<I', f.read(4))[0]
                    track.append((image_id, point2D_idx))

                points3D[point3D_id] = {
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'track': track
                }

        return points3D

    def count_matched_frames(self, verbose=False):
        """
        Count the number of frames that were successfully matched/registered by COLMAP.
        Returns (num_matched, total_input) tuple.
        """
        if not self.data_dir:
            if verbose:
                print("count_matched_frames: No data_dir set")
            return None, None

        # Find images.bin in COLMAP output
        images_bin_candidates = [
            os.path.join(self.data_dir, "colmap", "sparse", "0", "images.bin"),
            os.path.join(self.data_dir, "colmap", "sparse", "images.bin"),
            os.path.join(self.data_dir, "sparse", "0", "images.bin"),
            os.path.join(self.data_dir, "sparse", "images.bin"),
        ]

        images_bin_path = None
        for candidate in images_bin_candidates:
            if verbose:
                print(f"  Checking: {candidate} -> {os.path.exists(candidate)}")
            if os.path.exists(candidate):
                images_bin_path = candidate
                break

        if not images_bin_path:
            if verbose:
                print("count_matched_frames: No images.bin found")
            return None, None

        try:
            images = self.read_colmap_images_binary(images_bin_path)
            num_matched = len(images)
            if verbose:
                print(f"count_matched_frames: Found {num_matched} matched images")
        except Exception as e:
            print(f"Error reading images.bin: {e}")
            return None, None

        # Count total input frames
        images_dir = os.path.join(self.data_dir, "images")
        total_input = 0
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir):
                # Support various naming conventions
                if f.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
                    if not f.startswith("depth_"):
                        total_input += 1

        self.metrics['num_matched_frames'] = num_matched
        self.metrics['total_input_frames'] = total_input

        return num_matched, total_input

    def count_sparse_points(self, verbose=False):
        """
        Count the number of 3D points in the sparse point cloud.
        """
        if not self.data_dir:
            if verbose:
                print("count_sparse_points: No data_dir set")
            return None

        # Try PLY file first (faster)
        ply_candidates = [
            os.path.join(self.data_dir, "sparse_pc.ply"),
            os.path.join(self.data_dir, "colmap", "sparse", "0", "points3D.ply"),
            os.path.join(self.data_dir, "sparse", "0", "points3D.ply"),
            os.path.join(self.data_dir, "points3D.ply"),
            os.path.join(self.data_dir, "sparse_points.ply"),
        ]

        for ply_path in ply_candidates:
            if verbose:
                print(f"  Checking PLY: {ply_path} -> {os.path.exists(ply_path)}")
            if os.path.exists(ply_path) and HAS_OPEN3D:
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                    num_points = len(np.asarray(pcd.points))
                    self.metrics['num_sparse_points'] = num_points
                    if verbose:
                        print(f"count_sparse_points: Found {num_points} points in {ply_path}")
                    return num_points
                except Exception as e:
                    print(f"Error reading PLY: {e}")
                    continue

        # Fall back to reading COLMAP binary
        points_bin_candidates = [
            os.path.join(self.data_dir, "colmap", "sparse", "0", "points3D.bin"),
            os.path.join(self.data_dir, "colmap", "sparse", "points3D.bin"),
            os.path.join(self.data_dir, "sparse", "0", "points3D.bin"),
            os.path.join(self.data_dir, "sparse", "points3D.bin"),
        ]

        for points_path in points_bin_candidates:
            if verbose:
                print(f"  Checking BIN: {points_path} -> {os.path.exists(points_path)}")
            if os.path.exists(points_path):
                try:
                    points3D = self.read_colmap_points3D_binary(points_path)
                    num_points = len(points3D)
                    self.metrics['num_sparse_points'] = num_points
                    if verbose:
                        print(f"count_sparse_points: Found {num_points} points in {points_path}")
                    return num_points
                except Exception as e:
                    print(f"Error reading points3D.bin: {e}")
                    continue

        if verbose:
            print("count_sparse_points: No point cloud file found")
        return None

    def count_gaussians(self, verbose=False):
        """
        Count the number of Gaussians in the exported PLY file.
        The exported 3DGS PLY has one point per Gaussian.
        """
        if not self.data_dir:
            if verbose:
                print("count_gaussians: No data_dir set")
            return None

        # Check multiple possible locations and names
        splat_candidates = [
            os.path.join(self.data_dir, "gs_export", "splat.ply"),
            os.path.join(self.data_dir, "gs_export", "point_cloud.ply"),
            os.path.join(self.data_dir, "gs_export", "gaussian_splat.ply"),
            os.path.join(self.data_dir, "exports", "splat.ply"),
            os.path.join(self.data_dir, "splat.ply"),
            os.path.join(self.data_dir, "point_cloud.ply"),
        ]

        splat_path = None
        for candidate in splat_candidates:
            if verbose:
                print(f"  Checking: {candidate} -> {os.path.exists(candidate)}")
            if os.path.exists(candidate):
                splat_path = candidate
                break

        if not splat_path:
            if verbose:
                print("count_gaussians: No exported PLY found")
            return None

        # Try reading the PLY header first (works without Open3D and handles 3DGS format better)
        try:
            with open(splat_path, 'rb') as f:
                while True:
                    line = f.readline()
                    if b'end_header' in line:
                        break
                    if b'element vertex' in line:
                        parts = line.decode('ascii').strip().split()
                        num_gaussians = int(parts[2])
                        self.metrics['num_gaussians'] = num_gaussians
                        if verbose:
                            print(f"count_gaussians: Found {num_gaussians} gaussians (from header)")
                        return num_gaussians
        except Exception as e:
            if verbose:
                print(f"Error parsing PLY header: {e}")

        # Fallback to Open3D
        if HAS_OPEN3D:
            try:
                pcd = o3d.io.read_point_cloud(splat_path)
                num_gaussians = len(np.asarray(pcd.points))
                self.metrics['num_gaussians'] = num_gaussians
                if verbose:
                    print(f"count_gaussians: Found {num_gaussians} gaussians (from Open3D)")
                return num_gaussians
            except Exception as e:
                print(f"Error reading splat PLY: {e}")

        return None

    def calculate_psnr_ssim(self, config_path=None):
        """
        Calculate PSNR and SSIM using Nerfstudio's ns-eval command.
        Returns (psnr, ssim) tuple or (None, None) if evaluation fails.

        Note: This requires a trained model with config.yml
        """
        if not self.data_dir:
            return None, None

        # Find config.yml if not provided
        if config_path is None:
            outputs_dir = os.path.join(self.data_dir, "outputs")
            if os.path.exists(outputs_dir):
                for root, dirs, files in os.walk(outputs_dir):
                    if "config.yml" in files:
                        config_path = os.path.join(root, "config.yml")
                        break

        if not config_path or not os.path.exists(config_path):
            print("No config.yml found for evaluation")
            return None, None

        # Create output directory for eval results
        eval_output_dir = os.path.join(self.data_dir, "eval_output")
        os.makedirs(eval_output_dir, exist_ok=True)

        output_json = os.path.join(eval_output_dir, "metrics.json")

        cmd = [
            "ns-eval",
            "--load-config", config_path,
            "--output-path", output_json
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.data_dir,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"ns-eval failed: {result.stderr}")
                return None, None

            # Parse the output JSON
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    eval_results = json.load(f)

                # Nerfstudio eval output format
                psnr = eval_results.get('results', {}).get('psnr', None)
                ssim = eval_results.get('results', {}).get('ssim', None)

                # Try alternate key paths
                if psnr is None:
                    psnr = eval_results.get('psnr', None)
                if ssim is None:
                    ssim = eval_results.get('ssim', None)

                self.metrics['psnr'] = psnr
                self.metrics['ssim'] = ssim

                return psnr, ssim

        except subprocess.TimeoutExpired:
            print("ns-eval timed out")
            return None, None
        except Exception as e:
            print(f"Error running ns-eval: {e}")
            return None, None

        return None, None

    def calculate_all_metrics(self, config_path=None, skip_eval=False, verbose=False):
        """
        Calculate all available metrics.

        Args:
            config_path: Optional path to config.yml for PSNR/SSIM evaluation
            skip_eval: If True, skip PSNR/SSIM calculation (faster)
            verbose: If True, print detailed debug information

        Returns dict with all metrics.
        """
        self.metrics['timestamp'] = datetime.now().isoformat()

        if verbose:
            print(f"\n=== Calculating metrics for: {self.data_dir} ===")

        # Always calculate these (fast)
        self.count_matched_frames(verbose=verbose)
        self.count_sparse_points(verbose=verbose)
        self.count_gaussians(verbose=verbose)

        # PSNR/SSIM is slow, only calculate if requested
        if not skip_eval:
            self.calculate_psnr_ssim(config_path)

        if verbose:
            print(f"\nResults: {self.metrics}")

        return self.metrics.copy()

    def export_to_csv(self, output_path=None):
        """
        Export metrics to a CSV file.

        Args:
            output_path: Optional path for CSV file.
                        Defaults to gs_export/metrics.csv in data_dir.
        """
        if not self.data_dir:
            return None

        if output_path is None:
            export_dir = os.path.join(self.data_dir, "gs_export")
            os.makedirs(export_dir, exist_ok=True)
            output_path = os.path.join(export_dir, "metrics.csv")

        # Prepare row data
        row = {
            'timestamp': self.metrics.get('timestamp', datetime.now().isoformat()),
            'capture_name': os.path.basename(self.data_dir),
            'psnr': self.metrics.get('psnr', ''),
            'ssim': self.metrics.get('ssim', ''),
            'num_gaussians': self.metrics.get('num_gaussians', ''),
            'num_sparse_points': self.metrics.get('num_sparse_points', ''),
            'num_matched_frames': self.metrics.get('num_matched_frames', ''),
            'total_input_frames': self.metrics.get('total_input_frames', ''),
        }

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(output_path)

        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"Metrics exported to: {output_path}")
        return output_path

    def get_metrics_summary(self):
        """
        Get a formatted string summary of all metrics.
        """
        lines = ["=== Pipeline Metrics ==="]

        if self.metrics.get('num_matched_frames') is not None:
            total = self.metrics.get('total_input_frames', '?')
            lines.append(f"Matched Frames: {self.metrics['num_matched_frames']}/{total}")

        if self.metrics.get('num_sparse_points') is not None:
            lines.append(f"Sparse Points: {self.metrics['num_sparse_points']:,}")

        if self.metrics.get('num_gaussians') is not None:
            lines.append(f"Gaussians: {self.metrics['num_gaussians']:,}")

        if self.metrics.get('psnr') is not None:
            lines.append(f"PSNR: {self.metrics['psnr']:.2f} dB")

        if self.metrics.get('ssim') is not None:
            lines.append(f"SSIM: {self.metrics['ssim']:.4f}")

        if len(lines) == 1:
            lines.append("No metrics available yet")

        return "\n".join(lines)

    def get_metrics(self):
        """Return the current metrics dictionary"""
        return self.metrics

    def export_metrics(self):
        """Alias for export_to_csv to support GUI"""
        return self.export_to_csv()


# Standalone usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        calc = MetricsCalculator(data_dir)
        metrics = calc.calculate_all_metrics(skip_eval=True)
        print(calc.get_metrics_summary())
        calc.export_to_csv()
    else:
        print("Usage: python metrics_calculator.py <capture_directory>")
