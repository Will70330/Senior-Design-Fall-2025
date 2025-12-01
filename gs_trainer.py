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

        # Determine PLY for initialization
        ply_file = "object_final.ply" if use_masked else "map_final.ply"
        ply_path = os.path.join(data_dir, ply_file)
        
        if not os.path.exists(ply_path):
            # Try alternate
            alt_ply = "map_final.ply" if use_masked else "object_final.ply"
            if os.path.exists(os.path.join(data_dir, alt_ply)):
                ply_path = os.path.join(data_dir, alt_ply)
            else:
                ply_path = None

        print(f"Starting 3DGS Training using {config_file}...")
        if ply_path:
            print(f"Initializing from Point Cloud: {os.path.basename(ply_path)}")
        else:
            print("No initial point cloud found. Using random initialization.")
            
        print("Navigate to https://viewer.nerf.studio (or localhost link printed below) to watch.")
        
        data_path = os.path.join(data_dir, config_file)
        
        cmd = [
            "ns-train", "splatfacto",
            "--data", data_path,
            "--viewer.websocket-port", "7007",
            "--vis", "viewer"
        ]
        
        if ply_path:
            cmd.extend(["--pipeline.model.ply-file-path", ply_path])
        
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

    def export_ply(self, config_path, output_path):
        """
        Runs ns-export to generate the PLY file.
        config_path: Path to the config.yml generated during training.
        """
        if not self.is_installed():
            return
            
        print(f"Exporting PLY to {output_path}...")
        cmd = [
            "ns-export", "gaussian-splat",
            "--load-config", config_path,
            "--output-dir", os.path.dirname(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Export complete.")
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
