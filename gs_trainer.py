import subprocess
import os
import sys
import threading
import time
import shutil

class GsTrainer:
    def __init__(self):
        self.process = None
        self.training = False
        
    def is_installed(self):
        """Checks if ns-train is available in the path."""
        return shutil.which("ns-train") is not None

    def train(self, data_dir, use_masked=True, output_dir=None):
        """
        Starts Nerfstudio training in a background process.
        """
        if self.training:
            print("Training already in progress.")
            return

        if not self.is_installed():
            print("Error: Nerfstudio is not installed.")
            print("Please install it: pip install nerfstudio")
            return

        # Determine config
        config_file = "transforms_masked.json" if use_masked else "transforms.json"
        if not os.path.exists(os.path.join(data_dir, config_file)):
            print(f"Error: Data file {config_file} not found in {data_dir}")
            if use_masked and os.path.exists(os.path.join(data_dir, "transforms.json")):
                print("Falling back to transforms.json")
                config_file = "transforms.json"
                use_masked = False # Fallback
            else:
                return

        # Check for point cloud files (prioritize sparse_points.ply for nerfstudio)
        ply_candidates = [
            "sparse_points.ply",  # Nerfstudio expects this from COLMAP/SFM
            "object_final.ply" if use_masked else "map_final.ply",
            "map_final.ply" if use_masked else "object_final.ply"
        ]

        ply_path = None
        for ply_file in ply_candidates:
            candidate_path = os.path.join(data_dir, ply_file)
            if os.path.exists(candidate_path):
                ply_path = candidate_path
                break

        print(f"Starting 3DGS Training using {config_file}...")
        if ply_path:
            ply_name = os.path.basename(ply_path)
            print(f"Point Cloud found: {ply_name}")
            if ply_name == "sparse_points.ply":
                print("Nerfstudio will initialize Gaussians from sparse_points.ply")
            else:
                print(f"Note: Nerfstudio may use {ply_name} or random initialization")
        else:
            print("No point cloud found. Nerfstudio will use random initialization.")

        print("Navigate to https://viewer.nerf.studio (or localhost link printed below) to watch.")

        data_path = os.path.join(data_dir, config_file)

        cmd = [
            "ns-train", "splatfacto",
            "--data", data_path,
            "--viewer.websocket-port", "7007",
            "--vis", "viewer"
        ]

        # Note: In nerfstudio 1.1.5, PLY initialization via command line is not supported
        # The model will use random initialization or SFM points from the data

        if output_dir:
            cmd.extend(["--output-dir", output_dir])

        try:
            # Run in separate process group so we can kill it cleanly
            self.process = subprocess.Popen(
                cmd, 
                cwd=data_dir,
                stdout=sys.stdout, 
                stderr=sys.stderr,
                preexec_fn=os.setsid
            )
            self.training = True
            
            # Start a monitor thread
            threading.Thread(target=self._monitor_process, daemon=True).start()
            
        except Exception as e:
            print(f"Failed to start training: {e}")
            self.training = False

    def _monitor_process(self):
        if self.process:
            self.process.wait()
            self.training = False
            print("\nTraining process ended.")

    def stop(self):
        if self.process and self.training:
            print("Stopping training...")
            try:
                os.killpg(os.getpgid(self.process.pid), 15) # SIGTERM
            except Exception:
                pass
            self.process = None
            self.training = False

    def export_ply(self, config_path, output_path, cwd=None):
        """
        Runs ns-export to generate the PLY file.
        config_path: Path to the config.yml generated during training.
        cwd: Directory to run the command from (should match training CWD).
        """
        if not self.is_installed():
            return
            
        print(f"Exporting PLY to {output_path}...")
        cmd = [
            "ns-export", "gaussian-splat",
            "--load-config", config_path,
            "--output-dir", os.path.dirname(output_path)
        ]
        
        # Use provided CWD or default to config directory (which is often wrong for ns-export)
        # If not provided, we try to be smart: if config is deep in outputs/, assume CWD is the project root relative to it.
        # But best is to pass it.
        
        run_dir = cwd if cwd else os.path.dirname(config_path)
        
        try:
            subprocess.run(cmd, check=True, cwd=run_dir)
            print("Export complete.")
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
