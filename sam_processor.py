import cv2
import numpy as np
import os
import glob
import json
import sys
import open3d as o3d
import copy

# Helper to download model if needed
def download_sam_checkpoint():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    path = "sam_vit_h_4b8939.pth"
    expected_size = 2560000000 # Approx 2.56 GB
    
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            sys.stdout.write(f"\rDownloading SAM Checkpoint: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)")
            sys.stdout.flush()

    if os.path.exists(path):
        # Basic check: If file is too small (< 2GB), it's likely corrupt/partial
        size = os.path.getsize(path)
        if size < 2000000000:
            print(f"\nExisting checkpoint seems corrupt (size {size/1024/1024:.1f} MB). Re-downloading...")
            os.remove(path)
        else:
            return path

    print(f"Downloading SAM Checkpoint to {path}...")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, path, reporthook=report_hook)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(path):
            os.remove(path)
        raise e
        
    return path

class SamProcessor:
    def __init__(self, recording_dir, intrinsics_o3d):
        self.recording_dir = recording_dir
        self.images_dir = os.path.join(recording_dir, "images")
        self.intrinsics_o3d = intrinsics_o3d
        self.device = "cuda" # Assume CUDA if installing pytorch
        
        # Load Dependencies
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            self.torch = torch
            self.sam_model_registry = sam_model_registry
            self.SamPredictor = SamPredictor
            self.SamAutomaticMaskGenerator = SamAutomaticMaskGenerator
            
            if not torch.cuda.is_available():
                print("Warning: CUDA not found, using CPU. SAM will be slow.")
                self.device = "cpu"
        except ImportError:
            raise ImportError("Missing dependencies: torch, segment-anything")

    def load_trajectory(self):
        traj_path = os.path.join(self.recording_dir, "trajectory.txt")
        poses = {} # frame_idx -> 4x4 matrix
        if not os.path.exists(traj_path):
            return poses
            
        with open(traj_path, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 18: continue
                idx = int(parts[0])
                mat = np.array([float(x) for x in parts[2:]]).reshape(4, 4)
                poses[idx] = mat
        return poses

    def get_image_files(self):
        return sorted(glob.glob(os.path.join(self.images_dir, "frame_*.jpg")))

    def get_depth_files(self):
        return sorted(glob.glob(os.path.join(self.images_dir, "depth_*.png")))

    def run(self):
        # 1. Setup
        checkpoint_path = download_sam_checkpoint()
        print("Loading SAM model...")
        
        try:
            sam = self.sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        except Exception as e:
            print(f"\nError loading SAM model: {e}")
            print("The checkpoint file appears to be corrupt. Deleting it.")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            print("Please run the command again to re-download.")
            raise e

        sam.to(device=self.device)
        predictor = self.SamPredictor(sam)
        mask_generator = self.SamAutomaticMaskGenerator(sam)
        
        image_files = self.get_image_files()
        depth_files = self.get_depth_files()
        poses = self.load_trajectory()
        
        if not image_files:
            print("No images found.")
            return

        # 2. User Input (First Frame)
        first_img = cv2.imread(image_files[0])
        first_depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
        
        points = []
        labels = []
        
        # UI State
        ui = {
            'brush_size': 20,
            'drawing': False,
            'cur_label': 1, # 1: FG, 0: BG
            'mouse_pos': (0, 0),
            'mode': 'manual', # 'manual' or 'auto_select'
            'auto_masks': [],
            'selected_mask_indices': set(),
            'mask_colors': None
        }

        def update_display():
            if ui['mode'] == 'auto_select':
                # Create base image
                display = first_img.copy()
                
                # Ensure colors exist
                if ui['mask_colors'] is None and ui['auto_masks']:
                    np.random.seed(42)
                    ui['mask_colors'] = np.random.randint(0, 255, (len(ui['auto_masks']), 3), dtype=np.uint8)

                # Draw masks
                if ui['auto_masks']:
                    # We'll create a composite mask layer
                    # Unselected: Alpha 0.3
                    # Selected: Alpha 0.7 + White Border?
                    
                    overlay = display.copy()
                    
                    # Sort for drawing order (doesn't matter much for blending, but consistent is good)
                    # Draw unselected first
                    for i, mask_data in enumerate(ui['auto_masks']):
                        if i in ui['selected_mask_indices']: continue
                        
                        m = mask_data['segmentation']
                        color = ui['mask_colors'][i]
                        overlay[m] = overlay[m] * 0.4 + color * 0.6
                    
                    # Draw selected on top
                    for i in ui['selected_mask_indices']:
                        if i >= len(ui['auto_masks']): continue
                        mask_data = ui['auto_masks'][i]
                        m = mask_data['segmentation']
                        color = ui['mask_colors'][i]
                        # Make selected very bright/solid
                        overlay[m] = overlay[m] * 0.1 + color * 0.9
                        
                        # Draw contour for selected
                        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

                    # Blend overlay
                    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
                
                cv2.putText(display, "AUTO MODE: Click to select/deselect parts.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, "Press [SPACE] to Confirm Selection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Select Object", display)
                
            else:
                img = first_img.copy()
                # Draw existing points
                for pt, lb in zip(points, labels):
                    color = (0, 255, 0) if lb == 1 else (0, 0, 255)
                    cv2.circle(img, tuple(pt), 3, color, -1)
                
                # Draw brush cursor
                mx, my = ui['mouse_pos']
                cursor_color = (0, 255, 0) if ui['cur_label'] == 1 else (0, 0, 255)
                if not ui['drawing']:
                    cursor_color = (255, 255, 255) # White when hovering
                
                cv2.circle(img, (mx, my), ui['brush_size'], cursor_color, 1)
                cv2.imshow("Select Object", img)

        def mouse_callback(event, x, y, flags, param):
            ui['mouse_pos'] = (x, y)
            
            if ui['mode'] == 'auto_select':
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Find which mask was clicked (smallest area first)
                    sorted_indices = sorted(range(len(ui['auto_masks'])), key=lambda i: ui['auto_masks'][i]['area'])
                    
                    clicked_idx = -1
                    for idx in sorted_indices:
                        m = ui['auto_masks'][idx]
                        if m['segmentation'][y, x]:
                            clicked_idx = idx
                            break
                    
                    if clicked_idx != -1:
                        # Toggle selection
                        if clicked_idx in ui['selected_mask_indices']:
                            ui['selected_mask_indices'].remove(clicked_idx)
                        else:
                            ui['selected_mask_indices'].add(clicked_idx)
                        update_display()

            else:
                if event == cv2.EVENT_LBUTTONDOWN:
                    ui['drawing'] = True
                    ui['cur_label'] = 1
                    points.append([x, y])
                    labels.append(1)
                elif event == cv2.EVENT_RBUTTONDOWN:
                    ui['drawing'] = True
                    ui['cur_label'] = 0
                    points.append([x, y])
                    labels.append(0)
                elif event == cv2.EVENT_MOUSEMOVE:
                    if ui['drawing']:
                        if points:
                            last_pt = points[-1]
                            dist = np.hypot(x - last_pt[0], y - last_pt[1])
                            if dist > max(5, ui['brush_size'] / 4): 
                                points.append([x, y])
                                labels.append(ui['cur_label'])
                elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                    ui['drawing'] = False
                
                update_display()

        cv2.namedWindow("Select Object")
        cv2.setMouseCallback("Select Object", mouse_callback)
        
        print("\n--- Selection Controls ---")
        print("  Left Drag   : Paint Foreground (Green)")
        print("  Right Drag  : Paint Background (Red)")
        print("  [A]         : Auto-Segment Mode")
        print("  [ ] or [[]  : Adjust Brush Size")
        print("  [C]         : Clear Selection")
        print("  [SPACE]     : Finish and Process")
        print("--------------------------\n")
        
        update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # Space
                if ui['mode'] == 'auto_select':
                    # Commit auto selection
                    print(f"Committing {len(ui['selected_mask_indices'])} segments...")
                    for idx in ui['selected_mask_indices']:
                        m = ui['auto_masks'][idx]
                        ys, xs = np.where(m['segmentation'])
                        if len(xs) > 0:
                            # Sample points (keep it reasonable, e.g., 50 per segment)
                            num_pts = min(len(xs), 50)
                            indices = np.random.choice(len(xs), num_pts, replace=False)
                            for i in indices:
                                points.append([xs[i], ys[i]])
                                labels.append(1)
                    
                    ui['mode'] = 'manual'
                    ui['selected_mask_indices'].clear()
                    update_display()
                else:
                    # Finish manual selection
                    break
            elif key == 27: # Esc
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                points.clear()
                labels.clear()
                if ui['mode'] == 'auto_select':
                    ui['selected_mask_indices'].clear()
                update_display()
            elif key == ord(']'):
                ui['brush_size'] = min(200, ui['brush_size'] + 5)
                update_display()
            elif key == ord('['):
                ui['brush_size'] = max(5, ui['brush_size'] - 5)
                update_display()
            elif key == ord('a'):
                if ui['mode'] == 'manual':
                    if not ui['auto_masks']:
                        print("Running Auto-Segment... (this may take a moment)")
                        sys.stdout.flush()
                        masks = mask_generator.generate(first_img)
                        ui['auto_masks'] = masks
                        ui['mask_colors'] = None # Will init in update_display
                    
                    ui['mode'] = 'auto_select'
                    ui['selected_mask_indices'].clear()
                    update_display()
                    print("Auto-Segment Mode: Click objects to select. Space to confirm.")
                else:
                    # Cancel auto mode
                    ui['mode'] = 'manual'
                    update_display()

        cv2.destroyWindow("Select Object")
        
        if not points:
            print("No points selected.")
            return

        # 3. Processing Loop
        output_dir = os.path.join(self.recording_dir, "masked")
        os.makedirs(output_dir, exist_ok=True)
        
        # Tracking State: 3D Point Cloud of the object
        object_pcd = o3d.geometry.PointCloud()
        
        # Intrinsics matrix
        K = self.intrinsics_o3d.intrinsic_matrix
        
        for i, (img_path, depth_path) in enumerate(zip(image_files, depth_files)):
            print(f"Processing frame {i}/{len(image_files)}...")
            image = cv2.imread(img_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            # Prepare prompts
            current_points = []
            current_labels = []
            
            if i == 0:
                # Use user clicks
                current_points = np.array(points)
                current_labels = np.array(labels)
            else:
                # Project 3D Object Cloud to this frame
                if len(object_pcd.points) > 0:
                    # Transform world points to current camera frame
                    # T_cam_world = inv(T_world_cam)
                    if i not in poses:
                        print(f"Skipping frame {i}: No pose found.")
                        continue
                        
                    pose = poses[i]
                    pose_inv = np.linalg.inv(pose)
                    
                    # Transform points
                    pcd_local = copy.deepcopy(object_pcd)
                    pcd_local.transform(pose_inv)
                    pts_local = np.asarray(pcd_local.points)
                    
                    # Project to 2D
                    # x = K * X
                    pts_2d = K @ pts_local.T
                    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
                    pts_2d = pts_2d.T # (N, 2)
                    
                    # Filter points inside image bounds
                    h, w = image.shape[:2]
                    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h) & \
                           (pts_local[:, 2] > 0) # Z > 0
                    
                    valid_pts = pts_2d[mask]
                    
                    if len(valid_pts) > 0:
                        # Subsample points to use as prompts (max 5-10)
                        # Prefer points close to the center of the previous mask? 
                        # Just random sample for coverage
                        if len(valid_pts) > 5:
                            indices = np.random.choice(len(valid_pts), 5, replace=False)
                            valid_pts = valid_pts[indices]
                        
                        current_points = valid_pts
                        current_labels = np.ones(len(current_points)) # All foreground
                    else:
                        print("Tracking lost (no projected points).")
                        # Fallback: Use center? Or skip?
                        continue
            
            # Run SAM
            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if len(current_points) > 0:
                masks, scores, logits = predictor.predict(
                    point_coords=current_points,
                    point_labels=current_labels,
                    multimask_output=False
                )
                best_mask = masks[0]
                
                # Save Masked Image
                masked_img = image.copy()
                masked_img[~best_mask] = 0 # Black background
                
                cv2.imwrite(os.path.join(output_dir, f"masked_{i:05d}.jpg"), masked_img)
                
                # Update 3D Object Model (Accumulate points)
                # Only update every few frames to avoid massive cloud
                if i % 5 == 0:
                    # Back-project mask to 3D
                    # Create partial PCD
                    depth_masked = depth.copy()
                    depth_masked[~best_mask] = 0
                    
                    d_o3d = o3d.geometry.Image(depth_masked)
                    c_o3d = o3d.geometry.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        c_o3d, d_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
                    )
                    partial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics_o3d)
                    
                    # Transform to World
                    if i in poses:
                        partial_pcd.transform(poses[i])
                        # Add to accumulator
                        object_pcd += partial_pcd
                        object_pcd = object_pcd.voxel_down_sample(voxel_size=0.02) # Keep it clean
            else:
                print(f"No prompts for frame {i}")

        print("Processing Complete.")
        # Save the full object point cloud
        o3d.io.write_point_cloud(os.path.join(self.recording_dir, "object_final.ply"), object_pcd)
        
        # Generate transforms_masked.json for Nerfstudio
        self.save_masked_transforms(poses)

    def save_masked_transforms(self, poses):
        print("Generating transforms_masked.json...")
        
        # Basic intrinsics
        K = self.intrinsics_o3d.intrinsic_matrix
        out_data = {
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "w": self.intrinsics_o3d.width,
            "h": self.intrinsics_o3d.height,
            "camera_model": "OPENCV",
            "frames": []
        }
        
        # Sort indices
        indices = sorted(poses.keys())
        
        for idx in indices:
            # Check if masked image exists
            masked_path = os.path.join("masked", f"masked_{idx:05d}.jpg")
            full_path = os.path.join(self.recording_dir, masked_path)
            
            if not os.path.exists(full_path):
                continue
                
            pose = poses[idx]
            out_data["frames"].append({
                "file_path": masked_path,
                "transform_matrix": pose.tolist()
            })
            
        out_path = os.path.join(self.recording_dir, "transforms_masked.json")
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=4)
            
        print(f"Saved {out_path} with {len(out_data['frames'])} frames.")
