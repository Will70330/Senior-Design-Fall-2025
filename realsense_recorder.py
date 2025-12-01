import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import open3d as o3d
import copy
import json
from abc import ABC, abstractmethod
from gs_trainer import GsTrainer

# Try to import SAM processor (will be created later)
try:
    from sam_processor import SamProcessor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

class PoseEstimator(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, color_image, depth_image, intrinsics):
        pass

class Open3DVisualOdometry(PoseEstimator):
    def __init__(self):
        self.odo_init = np.identity(4)
        self.prev_rgbd = None
        # Align Camera (Y-down) with World (Y-up)
        self.flip_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        self.cur_pose = self.flip_transform.copy()
        self.option = o3d.pipelines.odometry.OdometryOption()

    def reset(self):
        self.odo_init = np.identity(4)
        self.prev_rgbd = None
        self.cur_pose = self.flip_transform.copy()

    def update(self, color_image, depth_image, intrinsics):
        color = o3d.geometry.Image(color_image)
        depth = o3d.geometry.Image(depth_image)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, 
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=True
        )

        if self.prev_rgbd is None:
            self.prev_rgbd = rgbd
            return self.cur_pose

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd, self.prev_rgbd, intrinsics, self.odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.option
        )

        if success:
            self.cur_pose = np.dot(self.cur_pose, trans)
            self.prev_rgbd = rgbd
            return self.cur_pose
        else:
            print("Odometry tracking lost!")
            return None

class RealSenseRecorder:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        ctx = rs.context()
        if len(ctx.devices) > 0:
            dev = ctx.devices[0]
            print(f"Device Name: {dev.get_info(rs.camera_info.name)}")
        else:
            print("No device connected")

        width, height = 640, 480
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        try:
            self.config.enable_stream(rs.stream.accel)
            self.config.enable_stream(rs.stream.gyro)
        except Exception:
            pass

        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        
        # State
        self.streaming = False
        self.recording = False
        self.show_pc_overlay = False
        self.mapping_enabled = False
        self.decimation_percentage = 10
        self.target_map_fps = 5
        self.record_dir = ""
        self.frame_idx = 0
        self.scene_mode = "Object" # "Large" or "Object"
        
        self.last_points = None
        self.last_color_image = None

        self.pose_estimator = Open3DVisualOdometry()
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, width, height, width / 2, height / 2
        )
        self.intrinsics_set = False
        self.current_pose = self.pose_estimator.cur_pose
        
        # Mapping
        self.global_map = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis_geometry_ptr = None
        self.map_initialized = False
        
        # 3DGS Trainer
        self.gs_trainer = GsTrainer()

        self.window_name = "RealSense Recorder"

    def on_fps_trackbar(self, val):
        self.target_map_fps = max(1, val)

    def start(self):
        try:
            self.profile = self.pipeline.start(self.config)
            self.streaming = True
            
            profile = self.profile.get_stream(rs.stream.color)
            intr = profile.as_video_stream_profile().get_intrinsics()
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )
            self.intrinsics_set = True
            print("Camera started.")
        except RuntimeError as e:
            print(f"Error starting camera: {e}")
            self.streaming = False

    def stop(self):
        if self.streaming:
            self.pipeline.stop()
            self.streaming = False
            print("Camera stopped.")
            if self.map_initialized:
                self.vis.destroy_window()
        # Stop trainer if running
        if self.gs_trainer.training:
            self.gs_trainer.stop()

    def create_recording_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = os.path.join(os.getcwd(), f"capture_{timestamp}")
        self.images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.imu_csv_path = os.path.join(self.record_dir, "imu_data.csv")
        with open(self.imu_csv_path, "w") as f:
            f.write("frame_idx,timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
        
        self.traj_path = os.path.join(self.record_dir, "trajectory.txt")
        
        self.reset_mapping()
        print(f"Recording to: {self.record_dir}")
        self.frame_idx = 0

    def reset_mapping(self):
        self.pose_estimator.reset()
        self.current_pose = self.pose_estimator.cur_pose
        
        if self.vis_geometry_ptr is not None:
            self.vis.remove_geometry(self.vis_geometry_ptr, reset_bounding_box=False)
            
        self.global_map.clear()
        self.vis.add_geometry(self.global_map, reset_bounding_box=False)
        self.vis_geometry_ptr = self.global_map
        print("Pose and Map reset.")

    def save_imu(self, accel, gyro, timestamp):
        if not self.recording or not self.record_dir:
            return
        
        # Handle cases where IMU data is missing
        ax, ay, az = (accel.x, accel.y, accel.z) if accel else (0, 0, 0)
        gx, gy, gz = (gyro.x, gyro.y, gyro.z) if gyro else (0, 0, 0)

        with open(self.imu_csv_path, "a") as f:
            f.write(f"{self.frame_idx},{timestamp:.3f},{ax},{ay},{az},{gx},{gy},{gz}\n")
            
    def save_pose(self, timestamp):
        with open(self.traj_path, "a") as f:
            flat_pose = self.current_pose.flatten()
            pose_str = ",".join([f"{x:.6f}" for x in flat_pose])
            f.write(f"{self.frame_idx},{timestamp},{pose_str}\n")

    def save_frame(self, color_image, depth_image):
        if not self.recording or not self.record_dir:
            return
        cv2.imwrite(os.path.join(self.images_dir, f"frame_{self.frame_idx:05d}.jpg"), color_image)
        cv2.imwrite(os.path.join(self.images_dir, f"depth_{self.frame_idx:05d}.png"), depth_image)
        self.frame_idx += 1

    def save_transforms_json(self):
        """Saves a transforms.json file compatible with Nerfstudio/Gaussian Splatting."""
        if not self.record_dir or not self.intrinsics_set:
            return
            
        print("Generating transforms.json...")
        frames = []
        # Read trajectory.txt
        if not os.path.exists(self.traj_path):
            print("No trajectory file found.")
            return

        with open(self.traj_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 18: continue
            
            idx = int(parts[0])
            # matrix is 16 floats starting at index 2
            matrix_flat = [float(x) for x in parts[2:]]
            matrix = np.array(matrix_flat).reshape(4, 4)
            
            # Convert to list
            transform_matrix = matrix.tolist()
            
            frames.append({
                "file_path": f"images/frame_{idx:05d}.jpg",
                "transform_matrix": transform_matrix
            })
            
        out_data = {
            "fl_x": self.o3d_intrinsics.intrinsic_matrix[0, 0],
            "fl_y": self.o3d_intrinsics.intrinsic_matrix[1, 1],
            "cx": self.o3d_intrinsics.intrinsic_matrix[0, 2],
            "cy": self.o3d_intrinsics.intrinsic_matrix[1, 2],
            "w": self.o3d_intrinsics.width,
            "h": self.o3d_intrinsics.height,
            "camera_model": "OPENCV",
            "frames": frames
        }
        
        with open(os.path.join(self.record_dir, "transforms.json"), "w") as f:
            json.dump(out_data, f, indent=4)
        print(f"Saved transforms.json with {len(frames)} frames.")

    def save_point_cloud(self):
        if not self.global_map.points or not self.record_dir:
            print("No global map data to save.")
            return

        print("Saving Global Point Cloud...")
        # We already have self.global_map which is an Open3D PointCloud
        # It is already in World coordinates
        
        ply_path = os.path.join(self.record_dir, "map_final.ply")
        o3d.io.write_point_cloud(ply_path, self.global_map)
        print(f"Saved global point cloud to {ply_path}")
        
    def update_global_map(self, color_image, depth_image):
        # 1. Create PCD
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        # Params based on scene mode
        depth_trunc = 1.5 if self.scene_mode == "Object" else 3.0
        voxel_size = 0.01 if self.scene_mode == "Object" else 0.05
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3d_intrinsics)
        
        # 2. Transform to World
        pcd.transform(self.current_pose)
        
        # 3. Update Map with robust visualizer handling
        if self.vis_geometry_ptr is not None:
            self.vis.remove_geometry(self.vis_geometry_ptr, reset_bounding_box=False)
        
        self.global_map += pcd
        self.global_map = self.global_map.voxel_down_sample(voxel_size=voxel_size)
        
        self.vis.add_geometry(self.global_map, reset_bounding_box=False)
        self.vis_geometry_ptr = self.global_map

    def toggle_scene_mode(self):
        if self.scene_mode == "Large":
            self.scene_mode = "Object"
        else:
            self.scene_mode = "Large"
        print(f"Scene Mode set to: {self.scene_mode}")
        # Reset map as voxel size changed
        self.reset_mapping()

    def select_recording_dir(self):
        # List capture directories
        base_dir = os.getcwd()
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("capture_")]
        dirs.sort(reverse=True) # Newest first
        
        if not dirs:
            print("No previous recordings found in current directory.")
            return None
            
        print("\nAvailable Recordings:")
        for i, d in enumerate(dirs):
            print(f"  [{i}] {d}")
        print("  [C] Cancel")
        
        try:
            choice = input("\nSelect recording to process (enter number): ").strip().lower()
            if choice == 'c':
                return None
            idx = int(choice)
            if 0 <= idx < len(dirs):
                return os.path.join(base_dir, dirs[idx])
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None

    def run_sam_post_process(self):
        if not SAM_AVAILABLE:
            print("Error: 'sam_processor' module not available.")
            print("Please install dependencies: pip install torch torchvision segment-anything")
            return

        # Determine directory
        target_dir = self.record_dir
        if not target_dir or not os.path.exists(target_dir):
            print("No active recording found. Please select one.")
            target_dir = self.select_recording_dir()
            
        if not target_dir:
            print("No directory selected. Aborting SAM processing.")
            return
            
        # Update state
        self.record_dir = target_dir
        
        print(f"Starting SAM Post-Processing on: {self.record_dir}")
        # Pause streaming to free resources
        was_streaming = self.streaming
        if self.streaming:
            self.stop()
            
        try:
            processor = SamProcessor(self.record_dir, self.o3d_intrinsics)
            processor.run()
        except Exception as e:
            print(f"SAM Processing failed: {e}")
            
        # Resume streaming
        if was_streaming:
            self.start()

    def run_training(self):
        if self.gs_trainer.training:
            self.gs_trainer.stop()
            return

        # Select Directory
        target_dir = self.record_dir
        if not target_dir or not os.path.exists(target_dir):
            print("No active recording found. Please select one.")
            target_dir = self.select_recording_dir()
        
        if not target_dir:
            return
            
        self.record_dir = target_dir
        
        # Check if masked data available
        has_masked = os.path.exists(os.path.join(target_dir, "transforms_masked.json"))
        if has_masked:
            print("Found masked dataset. Training on OBJECT.")
        else:
            print("No masks found. Training on WHOLE SCENE.")
            
        self.gs_trainer.train(target_dir, use_masked=has_masked)

    def run_gui(self):
        if not self.streaming:
            self.start()
            if not self.streaming:
                return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("PC Decimation %", self.window_name, 10, 100, lambda x: None) # lambda placeholder
        cv2.createTrackbar("Map FPS", self.window_name, 5, 30, self.on_fps_trackbar)
        
        self.vis.create_window(window_name="3D Map (Live)", width=800, height=600, left=700, top=50)
        self.vis.add_geometry(self.global_map)
        self.vis_geometry_ptr = self.global_map
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(axis)
        self.map_initialized = True

        print("Controls:")
        print("  [SPACE] : Start/Stop Recording")
        print("  [P]     : Toggle Point Cloud Overlay")
        print("  [M]     : Toggle Live 3D Mapping")
        print("  [R]     : Reset Pose Tracking")
        print("  [T]     : Toggle Scene Mode (Object/Large)")
        print("  [S]     : Run SAM Post-Processing")
        print("  [G]     : Train 3DGS (Nerfstudio)")
        print("  [Q]     : Quit")
        
        frame_count = 0

        while True:
            frames = self.pipeline.wait_for_frames()
            timestamp = frames.get_timestamp()
            frame_count += 1
            
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            if self.intrinsics_set:
                color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                new_pose = self.pose_estimator.update(color_rgb, depth_image, self.o3d_intrinsics)
                if new_pose is not None:
                    self.current_pose = new_pose
                    
            if self.recording:
                self.save_pose(timestamp)
                self.save_imu(None, None, timestamp) # Skipping IMU data extract for brevity/speed here
                self.save_frame(color_image, depth_image)
            
            skip_frames = max(1, int(30 / self.target_map_fps))
            if self.mapping_enabled and frame_count % skip_frames == 0:
                self.update_global_map(color_image, depth_image)
            
            self.vis.poll_events()
            self.vis.update_renderer()

            # Display
            display_image = color_image.copy()
            
            # HUD
            status_color = (0, 255, 0) if self.recording else (0, 0, 255)
            status_text = "RECORDING" if self.recording else "PAUSED"
            tx, ty, tz = self.current_pose[0, 3], self.current_pose[1, 3], self.current_pose[2, 3]
            
            cv2.putText(display_image, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(display_image, f"Pos: {tx:.2f}, {ty:.2f}, {tz:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_image, f"Map: {'ON' if self.mapping_enabled else 'OFF'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(display_image, f"Mode: {self.scene_mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.gs_trainer.training:
                cv2.putText(display_image, "TRAINING 3DGS...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(self.window_name, display_image)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                if self.recording:
                    self.recording = False
                    self.save_point_cloud()
                    self.save_transforms_json() # Save JSON at end
                    print("Recording stopped and data saved.")
                else:
                    self.create_recording_dir()
                    self.recording = True
                    print("Recording started...")
            elif key == ord('m'):
                self.mapping_enabled = not self.mapping_enabled
            elif key == ord('r'):
                self.reset_mapping()
            elif key == ord('t'):
                self.toggle_scene_mode()
            elif key == ord('s'):
                self.run_sam_post_process()
            elif key == ord('g'):
                self.run_training()

        self.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealSenseRecorder()
    app.run_gui()