#!/usr/bin/env python3
"""
Visualize COLMAP sparse 3D reconstruction
"""
import struct
import numpy as np
from pathlib import Path
import argparse

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Install with: pip install open3d")


def read_colmap_points3D_binary(path):
    """Read COLMAP points3D.bin file"""
    points3D = {}
    with open(path, "rb") as f:
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
                'error': error
            }

    return points3D


def visualize_sparse_colmap(sparse_dir, output_ply=None):
    """Visualize COLMAP sparse reconstruction"""
    sparse_path = Path(sparse_dir)
    points3D_path = sparse_path / "points3D.bin"

    if not points3D_path.exists():
        print(f"Error: {points3D_path} not found!")
        print(f"Looking for sparse reconstruction in: {sparse_path}")
        return

    # Read points
    points3D = read_colmap_points3D_binary(points3D_path)
    print(f"Loaded {len(points3D)} points")

    # Convert to arrays
    xyz = np.array([p['xyz'] for p in points3D.values()])
    rgb = np.array([p['rgb'] for p in points3D.values()]) / 255.0
    errors = np.array([p['error'] for p in points3D.values()])

    print(f"\nPoint cloud statistics:")
    print(f"  Number of points: {len(xyz)}")
    print(f"  Bounds: X=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}], "
          f"Y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}], "
          f"Z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    print(f"  Mean reprojection error: {errors.mean():.4f}")

    # Export to PLY if requested
    if output_ply:
        output_path = Path(output_ply)
        write_ply(output_path, xyz, rgb)
        print(f"\nExported to: {output_path}")

    # Visualize with Open3D if available
    if HAS_OPEN3D:
        print("\nOpening 3D viewer (close window to continue)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name="COLMAP Sparse Reconstruction",
            width=1920,
            height=1080,
        )
    else:
        print("\nTo visualize interactively, install Open3D:")
        print("  pip install open3d")


def write_ply(output_path, xyz, rgb):
    """Write point cloud to PLY format"""
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(len(xyz)):
            r, g, b = (rgb[i] * 255).astype(int)
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {r} {g} {b}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize COLMAP sparse reconstruction")
    parser.add_argument(
        "--sparse-dir",
        type=str,
        required=True,
        help="Path to COLMAP sparse directory (containing points3D.bin)"
    )
    parser.add_argument(
        "--output-ply",
        type=str,
        default=None,
        help="Optional: Export to PLY file"
    )

    args = parser.parse_args()
    visualize_sparse_colmap(args.sparse_dir, args.output_ply)


if __name__ == "__main__":
    main()
