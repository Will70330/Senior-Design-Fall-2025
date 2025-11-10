#!/usr/bin/env python3
"""
Interactive 3D Reconstruction Pipeline
COLMAP → Visualization → Training (NeRF/3DGS) → Metrics → Viewing
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import shutil

import viser
import numpy as np
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig


def print_header(text):
    """Print a nice header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def run_command(cmd, description, check=True):
    """Run a shell command with nice output"""
    print(f"→ {description}...")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"✗ Error: {result.stderr}")
        sys.exit(1)
    return result


def wait_for_user(message="Press Enter to continue, or 'q' to quit: "):
    """Wait for user input"""
    print(f"\n{'='*70}")
    response = input(message).strip().lower()
    if response in ['q', 'quit', 'exit']:
        print("Exiting...")
        sys.exit(0)
    return response


def read_ply(ply_path):
    """Read PLY point cloud"""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if 'red' in vertices.data.dtype.names:
        rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        rgb = np.ones_like(xyz) * 0.5

    return xyz, rgb


def visualize_colmap_in_viser(data_path, port=7007):
    """Visualize COLMAP reconstruction in Viser"""
    data_path = Path(data_path)

    # Load camera data
    print("Loading camera data...")
    config = NerfstudioDataParserConfig(data=data_path)
    dataparser = config.setup()
    dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
    cameras = dataparser_outputs.cameras

    print(f"Loaded {len(cameras)} cameras")

    # Load sparse point cloud
    ply_path = data_path / "sparse_pc.ply"
    if ply_path.exists():
        print(f"Loading sparse point cloud...")
        xyz, rgb = read_ply(ply_path)
        print(f"Loaded {len(xyz)} sparse points")
    else:
        print("Warning: No sparse point cloud found")
        xyz, rgb = None, None

    # Start Viser server
    print(f"\nStarting viewer on http://localhost:{port}")
    server = viser.ViserServer(port=port)

    # Add point cloud
    if xyz is not None:
        server.scene.add_point_cloud(
            name="/colmap_sparse",
            points=xyz,
            colors=rgb,
            point_size=0.01,
        )

    # Add cameras
    print("Adding cameras to viewer...")
    for i in range(len(cameras)):
        c2w = cameras.camera_to_worlds[i].cpu().numpy()

        fx = float(cameras.fx[i].cpu().numpy().item())
        width = float(cameras.width[i].cpu().numpy().item())
        height = float(cameras.height[i].cpu().numpy().item())

        server.scene.add_camera_frustum(
            name=f"/cameras/camera_{i:04d}",
            fov=fx / width,
            aspect=width / height,
            scale=0.1,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            color=(100, 200, 255),
        )

    # Add coordinate frame
    server.scene.add_frame(
        name="/world",
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
        axes_length=0.5,
        axes_radius=0.01,
    )

    print(f"\n{'='*70}")
    print(f"Viewer ready at: http://localhost:{port}")
    print(f"{'='*70}")
    print(f"Scene info:")
    print(f"  - {len(cameras)} camera poses (cyan frustums)")
    if xyz is not None:
        print(f"  - {len(xyz):,} sparse 3D points (colored)")
    print(f"  - World coordinate frame (RGB = XYZ)")
    print(f"{'='*70}\n")

    return server


def process_with_colmap(image_dir, output_dir):
    """Process images with COLMAP to create sparse reconstruction"""
    print_header("STEP 1: COLMAP Sparse Reconstruction")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        print(f"✗ Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Count images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.PNG")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG"))
    print(f"Found {len(image_files)} images in {image_dir}")

    if len(image_files) == 0:
        print(f"✗ Error: No images found in {image_dir}")
        sys.exit(1)

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    images_output = output_dir / "images"
    if not images_output.exists():
        print(f"Copying images to {images_output}...")
        shutil.copytree(image_dir, images_output)

    # Run NeRFStudio's data processing (which uses COLMAP with GPU)
    print(f"\nProcessing with ns-process-data (COLMAP with GPU acceleration)...")
    print("This may take a few minutes depending on the number of images...")

    cmd = f"ns-process-data images --data {images_output} --output-dir {output_dir}"
    result = run_command(cmd, "Running COLMAP feature extraction and matching", check=False)

    if result.returncode != 0:
        print("✗ COLMAP processing failed")
        return False

    print("✓ COLMAP processing complete!")

    # Check for outputs
    if (output_dir / "transforms.json").exists():
        print(f"✓ Camera transforms saved: {output_dir / 'transforms.json'}")

    if (output_dir / "sparse_pc.ply").exists():
        print(f"✓ Sparse point cloud saved: {output_dir / 'sparse_pc.ply'}")

    return True


def train_model(data_path, model_type, output_dir, iterations=30000):
    """Train NeRF or 3DGS model"""
    if model_type == "nerf":
        print_header("STEP 3: Training Instant-NGP (NeRF)")
        method = "instant-ngp"
    else:  # 3dgs
        print_header("STEP 3: Training 3D Gaussian Splatting")
        method = "splatfacto"

    print(f"Training with {iterations} iterations...")
    print("This will take several minutes to an hour depending on scene complexity.\n")

    model_output = output_dir / "model"

    cmd = (
        f"ns-train {method} "
        f"--data {data_path} "
        f"--max-num-iterations {iterations} "
        f"--viewer.quit-on-train-completion True "
        f"--output-dir {model_output}"
    )

    run_command(cmd, f"Training {method}", check=True)

    print(f"\n✓ Training complete!")

    # Find config file
    config_files = list(model_output.rglob("config.yml"))
    if config_files:
        config_path = config_files[0]
        print(f"✓ Model config: {config_path}")
        return config_path
    else:
        print("✗ Warning: Could not find config.yml")
        return None


def evaluate_model(config_path, output_dir):
    """Evaluate trained model and compute metrics"""
    print_header("Computing Metrics")

    eval_script = Path(__file__).parent / "eval_model.py"
    metrics_path = output_dir / "metrics.json"

    cmd = f"python {eval_script} --load-config {config_path} --output-path {metrics_path}"
    run_command(cmd, "Evaluating model", check=True)

    # Display metrics
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results = json.load(f)

        print(f"\n{'='*70}")
        print("METRICS:")
        print(f"{'='*70}")
        res = results.get("results", {})
        print(f"  PSNR:  {res.get('psnr', 'N/A'):.2f} dB")
        print(f"  SSIM:  {res.get('ssim', 'N/A'):.4f}")
        print(f"  FPS:   {res.get('fps', 'N/A'):.2f}")
        print(f"  LPIPS: {res.get('lpips', 'N/A'):.4f}")
        print(f"{'='*70}\n")

        print(f"✓ Metrics saved: {metrics_path}")

    return metrics_path


def render_video(config_path, output_dir):
    """Render video of the reconstruction"""
    print_header("Rendering Video")

    video_path = output_dir / "reconstruction_video.mp4"

    print("Rendering interpolated camera path...")

    cmd = (
        f"ns-render interpolate "
        f"--load-config {config_path} "
        f"--output-path {video_path} "
        f"--frame-rate 30"
    )

    result = run_command(cmd, "Rendering video", check=False)

    if result.returncode == 0 and video_path.exists():
        print(f"✓ Video saved: {video_path}")
        return video_path
    else:
        print("✗ Video rendering failed or not available")
        return None


def view_trained_model(config_path, port=7007):
    """Launch viewer for trained model"""
    print_header("Launching Viewer for Trained Model")

    print(f"Starting viewer on http://localhost:{port}")
    print("The viewer will open in your browser automatically.")
    print("Press Ctrl+C in the terminal to stop the viewer.\n")

    cmd = f"ns-viewer --load-config {config_path} --viewer.websocket-port {port}"

    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\nViewer closed.")
    except Exception as e:
        print(f"✗ Viewer failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Reconstruction Pipeline"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to input images directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/<scene_name>_<timestamp>)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Training iterations (default: 30000)"
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name = Path(args.images).name
        output_dir = Path("outputs") / f"{scene_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("Interactive 3D Reconstruction Pipeline")
    print(f"Input images: {args.images}")
    print(f"Output directory: {output_dir}")

    # Step 1: COLMAP processing
    success = process_with_colmap(args.images, output_dir)
    if not success:
        print("✗ COLMAP processing failed. Exiting.")
        sys.exit(1)

    # Step 2: Visualize sparse reconstruction
    print_header("STEP 2: Visualizing COLMAP Sparse Reconstruction")

    server = visualize_colmap_in_viser(output_dir, port=7007)

    # Wait for user
    wait_for_user("View the sparse reconstruction at http://localhost:7007\nPress Enter to continue: ")

    # Stop viewer
    print("Stopping viewer...")
    del server
    time.sleep(1)

    # Step 3: Choose model type
    print_header("Model Selection")
    print("Choose reconstruction method:")
    print("  1 - Instant-NGP (NeRF) - Fast, good quality")
    print("  2 - 3D Gaussian Splatting - Very fast rendering, excellent quality")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        model_type = "nerf"
    elif choice == "2":
        model_type = "3dgs"
    else:
        print("Invalid choice. Exiting.")
        sys.exit(0)

    # Step 4: Train model
    config_path = train_model(output_dir, model_type, output_dir, args.iterations)

    if not config_path:
        print("✗ Training failed. Exiting.")
        sys.exit(1)

    # Step 5: Evaluate model
    metrics_path = evaluate_model(config_path, output_dir)

    # Step 6: Render video
    video_path = render_video(config_path, output_dir)

    # Step 7: View trained model
    print_header("Final Results")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - Images: {output_dir / 'images'}")
    print(f"  - COLMAP data: {output_dir / 'colmap'}")
    print(f"  - Sparse point cloud: {output_dir / 'sparse_pc.ply'}")
    print(f"  - Transforms: {output_dir / 'transforms.json'}")
    print(f"  - Model: {output_dir / 'model'}")
    print(f"  - Config: {config_path}")
    print(f"  - Metrics: {metrics_path}")
    if video_path:
        print(f"  - Video: {video_path}")

    # Ask if user wants to view
    print(f"\n{'='*70}")
    response = input("Would you like to view the trained model? (y/n): ").strip().lower()

    if response in ['y', 'yes']:
        view_trained_model(config_path, port=7007)

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"To view the model later, run:")
    print(f"  ns-viewer --load-config {config_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
