#!/usr/bin/env python3
"""
View COLMAP sparse reconstruction using NeRFStudio's Viser viewer
"""
import argparse
from pathlib import Path
import time

import viser
import numpy as np
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.cameras import Cameras


def read_ply(ply_path):
    """Read PLY point cloud file"""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # Extract positions
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Extract colors if available
    if 'red' in vertices.data.dtype.names:
        rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        # Default gray color if no color info
        rgb = np.ones_like(xyz) * 0.5

    return xyz, rgb


def read_colmap_sparse(sparse_dir):
    """Read COLMAP sparse reconstruction"""
    import struct

    points3D_path = Path(sparse_dir) / "points3D.bin"

    if not points3D_path.exists():
        print(f"Error: {points3D_path} not found!")
        return None, None

    points3D = {}
    with open(points3D_path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        print(f"Reading {num_points} 3D points...")

        for i in range(num_points):
            point3D_id = struct.unpack("Q", f.read(8))[0]
            xyz = struct.unpack("ddd", f.read(24))
            rgb = struct.unpack("BBB", f.read(3))
            error = struct.unpack("d", f.read(8))[0]
            track_length = struct.unpack("Q", f.read(8))[0]
            track_elems = struct.unpack("ii" * track_length, f.read(8 * track_length))

            points3D[point3D_id] = {
                'xyz': np.array(xyz),
                'rgb': np.array(rgb),
            }

    xyz = np.array([p['xyz'] for p in points3D.values()])
    rgb = np.array([p['rgb'] for p in points3D.values()]) / 255.0

    return xyz, rgb


def view_colmap_data(data_path: str, port: int = 7007):
    """View COLMAP data with cameras and sparse point cloud"""
    data_path = Path(data_path)

    # Load camera data using NeRFStudio's dataparser
    print("Loading camera data...")
    config = NerfstudioDataParserConfig(data=data_path)
    dataparser = config.setup()
    dataparser_outputs = dataparser.get_dataparser_outputs(split="train")

    cameras = dataparser_outputs.cameras

    print(f"Loaded {len(cameras)} cameras")

    # Load sparse point cloud from PLY file (NeRFStudio format)
    ply_path = data_path / "sparse_pc.ply"
    if ply_path.exists():
        print(f"Loading sparse point cloud from {ply_path}...")
        xyz, rgb = read_ply(ply_path)
        print(f"Loaded {len(xyz)} sparse points")
    else:
        # Try COLMAP binary format
        sparse_dir = data_path / "sparse" / "0"
        if sparse_dir.exists():
            xyz, rgb = read_colmap_sparse(sparse_dir)
            if xyz is not None:
                print(f"Loaded {len(xyz)} sparse points")
        else:
            print(f"Warning: No sparse reconstruction found")
            xyz, rgb = None, None

    # Start Viser server
    print(f"\nStarting viewer on http://localhost:{port}")
    print("Press Ctrl+C to exit")
    server = viser.ViserServer(port=port)

    # Add point cloud
    if xyz is not None:
        print("Adding sparse point cloud to viewer...")
        server.scene.add_point_cloud(
            name="/colmap_sparse",
            points=xyz,
            colors=rgb,
            point_size=0.01,
        )

    # Add cameras
    print("Adding cameras to viewer...")
    for i in range(len(cameras)):
        # Get camera pose
        c2w = cameras.camera_to_worlds[i].cpu().numpy()

        # Extract scalar values properly
        fx = float(cameras.fx[i].cpu().numpy().item())
        width = float(cameras.width[i].cpu().numpy().item())
        height = float(cameras.height[i].cpu().numpy().item())

        # Add camera frustum
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

    print(f"\n{'='*60}")
    print(f"Viewer ready at: http://localhost:{port}")
    print(f"{'='*60}")
    print(f"Scene info:")
    print(f"  - {len(cameras)} camera poses (cyan frustums)")
    if xyz is not None:
        print(f"  - {len(xyz):,} sparse 3D points (colored)")
    print(f"  - World coordinate frame (RGB = XYZ)")
    print(f"\nControls:")
    print(f"  - Left click + drag: Rotate")
    print(f"  - Right click + drag: Pan")
    print(f"  - Scroll: Zoom")
    print(f"{'='*60}\n")

    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


def main():
    parser = argparse.ArgumentParser(
        description="View COLMAP reconstruction with NeRFStudio's Viser viewer"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., examples/datasets/nerfstudio/poster)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7007,
        help="Port for Viser server (default: 7007)"
    )

    args = parser.parse_args()
    view_colmap_data(args.data, args.port)


if __name__ == "__main__":
    main()
