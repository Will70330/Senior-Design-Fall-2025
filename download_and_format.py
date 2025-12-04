import os
import subprocess
import shutil
import sys
import argparse

# List of available Nerfstudio captures
AVAILABLE_DATASETS = [
    "bww_entrance", "campanile", "desolation", "library", "poster", 
    "redwoods2", "storefront", "vegetation", "Egypt", "person", 
    "kitchen", "plane", "dozer", "floating-tree", "aspen", "stump", 
    "sculpture", "Giannini-Hall"
]

def run_command(cmd):
    """Executes a shell command and returns the output."""
    print(f"Running: {cmd}")
    # We use check=False to handle errors manually, and remove capture_output to show progress bars
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download and format Nerfstudio sample datasets for the 3DGS GUI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    dataset_help = "Name of the dataset to download. Options:\n" + "\n".join(
        [f"  - {d}" for d in AVAILABLE_DATASETS]
    )
    
    parser.add_argument(
        "dataset", 
        nargs="?", 
        help=dataset_help,
        choices=AVAILABLE_DATASETS + ["all"]
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available datasets and exit."
    )

    args = parser.parse_args()

    if args.list:
        print("Available Datasets:")
        for d in AVAILABLE_DATASETS:
            print(f"  - {d}")
        return

    if not args.dataset:
        parser.print_help()
        print("\nError: Please specify a dataset name (e.g., 'poster').")
        return

    capture_name = args.dataset

    # 1. Define Paths
    current_dir = os.getcwd()
    download_root = os.path.join(current_dir, "data_download1")
    tmp_dir = os.path.join(current_dir, "tmp_download")
    
    # The final directory name the GUI will load
    target_capture_dir = os.path.join(download_root, f"capture_{capture_name}")
    target_images_dir = os.path.join(target_capture_dir, "images")

    # 2. Clean up previous runs
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if os.path.exists(target_capture_dir):
        print(f"Warning: Target directory {target_capture_dir} already exists.")
        ans = input("Overwrite? [y/N]: ").lower()
        if ans != 'y':
            print("Aborted.")
            return
        shutil.rmtree(target_capture_dir)

    os.makedirs(download_root, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # 3. Run ns-download-data
    print(f"Downloading dataset: {capture_name}...")
    cmd = f"ns-download-data nerfstudio --capture-name={capture_name} --save-dir {tmp_dir}"
    if not run_command(cmd):
        print("Failed to download data.")
        return

    # 4. Locate the downloaded data
    # Check various possible paths where the data might be
    possible_paths = [
        os.path.join(tmp_dir, "nerfstudio", capture_name), # tmp_download/nerfstudio/poster
        os.path.join(tmp_dir, capture_name),               # tmp_download/poster
        tmp_dir                                            # tmp_download/
    ]
    
    downloaded_path = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "images")) or os.path.exists(os.path.join(p, "images_4")):
            downloaded_path = p
            break
            
    if not downloaded_path:
        print(f"Error: Could not locate data. Searched: {possible_paths}")
        if os.path.exists(tmp_dir):
             # Recursive print
             print(f"Structure of {tmp_dir}:")
             for root, dirs, files in os.walk(tmp_dir):
                 level = root.replace(tmp_dir, '').count(os.sep)
                 indent = ' ' * 4 * (level)
                 print(f"{indent}{os.path.basename(root)}/")
                 subindent = ' ' * 4 * (level + 1)
                 for f in files:
                     print(f"{subindent}{f}")
        return
    
    print(f"Processing downloaded data from: {downloaded_path}")
    
    # Check what we got
    found_images = False
    source_images = None
    
    # Nerfstudio data usually has an 'images' folder
    # Prefer original resolution 'images', fall back to 'images_4' (downscaled)
    if os.path.exists(os.path.join(downloaded_path, "images")):
        source_images = os.path.join(downloaded_path, "images")
        found_images = True
    elif os.path.exists(os.path.join(downloaded_path, "images_4")):
        source_images = os.path.join(downloaded_path, "images_4")
        found_images = True
    
    if not found_images:
        print("Error: Could not find 'images' folder in downloaded data.")
        return

    # 5. Move and Format
    print(f"Formatting data into {target_capture_dir}...")
    os.makedirs(target_capture_dir, exist_ok=True)
    
    # Move images
    shutil.copytree(source_images, target_images_dir)
    print(f"Copied images to {target_images_dir}")

    # Copy transforms.json if it exists
    src_transforms = os.path.join(downloaded_path, "transforms.json")
    if os.path.exists(src_transforms):
        shutil.copy(src_transforms, os.path.join(target_capture_dir, "transforms.json"))
        print("Copied transforms.json")

    # Copy sparse pc if it exists (nerfstudio usually packages one)
    # It might be named 'sparse_pc.ply' or inside a colmap folder
    src_ply = os.path.join(downloaded_path, "sparse_pc.ply")
    if os.path.exists(src_ply):
        shutil.copy(src_ply, os.path.join(target_capture_dir, "sparse_pc.ply"))
        print("Copied sparse_pc.ply")
    else:
        # Check colmap folder
        src_colmap = os.path.join(downloaded_path, "colmap", "sparse", "0", "points3D.ply")
        if os.path.exists(src_colmap):
             shutil.copy(src_colmap, os.path.join(target_capture_dir, "sparse_points.ply"))
             print("Copied points3D.ply to sparse_points.ply")

    # Clean up tmp
    shutil.rmtree(tmp_dir)

    print("\nDone!")
    print(f"Dataset ready at: {target_capture_dir}")
    print("To use in GUI:")
    print("1. Click 'Folder Load Capture'")
    print(f"2. Select: {target_capture_dir}")
    print("3. You can now run '2. Generate Sparse PC' (if not included) or '3. Train 3DGS'")

if __name__ == "__main__":
    main()