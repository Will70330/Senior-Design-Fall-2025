import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import open3d as o3d
import copy
from abc import ABC, abstractmethod

class PoseEstimator(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, color_image, depth_image, intrinsics):
        """
        Updates the pose based on new frames.
        Returns: 4x4 transformation matrix (camera pose) or None if tracking lost.
        """
        pass

class Open3DVisualOdometry(PoseEstimator):
    def __init__(self):
        self.odo_init = np.identity(4)
        self.prev_rgbd = None
        self.cur_pose = np.identity(4)
        
        # Odometry config
        self.option = o3d.pipelines.odometry.OdometryOption()
        # We can tune these for performance vs accuracy
        # self.option.iteration_number_per_pyramid_level = o3d.utility.IntVector([20, 10, 5]) 

    def reset(self):
        self.odo_init = np.identity(4)
        self.prev_rgbd = None
        self.cur_pose = np.identity(4)

    def update(self, color_image, depth_image, intrinsics):
        # Create Open3D RGBD Image
        # Ensure images are contiguous and correct type for Open3D
        # Color: uint8, Depth: uint16
        color = o3d.geometry.Image(color_image)
        depth = o3d.geometry.Image(depth_image)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, 
            depth_scale=1000.0, # RealSense default is 1mm
            depth_trunc=3.0,    # Truncate at 3m for better odometry stability
            convert_rgb_to_intensity=True
        )

        if self.prev_rgbd is None:
            self.prev_rgbd = rgbd
            return self.cur_pose

        # Calculate Odometry
        # standard_pinhole_camera_intrinsic is expected
        # We assume intrinsics is an open3d.camera.PinholeCameraIntrinsic object
        
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd, self.prev_rgbd, intrinsics, self.odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.option
        )

        if success:
            # Update Pose: T_world_curr = T_world_prev * T_prev_curr
            # Open3D returns T_prev_curr (motion from prev to curr)
            # But wait, compute_rgbd_odometry returns the alignment from source (rgbd) to target (prev_rgbd).
            # It effectively calculates the pose of 'rgbd' relative to 'prev_rgbd'.
            # So 'trans' is T_prev_curr.
            
            # Update global pose
            self.cur_pose = np.dot(self.cur_pose, trans)
            self.prev_rgbd = rgbd
            return self.cur_pose
        else:
            # Tracking lost (or not enough features/movement)
            # For now, we just keep the previous pose and warn
            print("Odometry tracking lost!")
            return None

class RealSenseRecorder:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Diagnostic: Check device info
        ctx = rs.context()
        if len(ctx.devices) > 0:
            dev = ctx.devices[0]
            print(f"Device Name: {dev.get_info(rs.camera_info.name)}")
            print(f"USB Type: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
        else:
            print("No device connected")

        # Enable streams
        # Lower resolution to 640x480 for maximum compatibility (works on USB2 and USB3)
        width, height = 640, 480
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        # IMU streams (safely attempt to enable)
        try:
            self.config.enable_stream(rs.stream.accel)
            self.config.enable_stream(rs.stream.gyro)
        except Exception as e:
            print(f"Warning: Could not enable IMU streams: {e}")

        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        
        # State
        self.streaming = False
        self.recording = False
        self.show_pc_overlay = False
        self.mapping_enabled = True # Default ON
        self.decimation_percentage = 10  # Default 10%
        self.record_dir = ""
        self.frame_idx = 0
        
        # Point Cloud Data holders for saving
        self.last_points = None
        self.last_color_image = None # Store color image for coloring PC

        # Pose Estimation
        self.pose_estimator = Open3DVisualOdometry()
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, 
            width, height, # Fx, Fy ( placeholders, will update from camera)
            width / 2, height / 2 # Cx, Cy
        )
        self.intrinsics_set = False
        self.current_pose = np.identity(4)
        
        # Mapping (SLAM-ish)
        self.global_map = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.map_initialized = False

        # GUI Window Name
        self.window_name = "RealSense Recorder"

    def start(self):
        try:
            self.profile = self.pipeline.start(self.config)
            self.streaming = True
            
            # Get Intrinsics
            profile = self.profile.get_stream(rs.stream.color)
            intr = profile.as_video_stream_profile().get_intrinsics()
            
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )
            self.intrinsics_set = True
            
            print("Camera started.")
        except RuntimeError as e:
            print(f"Error starting camera: {e}")
            print("Is the device connected?")
            self.streaming = False

    def stop(self):
        if self.streaming:
            self.pipeline.stop()
            self.streaming = False
            print("Camera stopped.")
            if self.map_initialized:
                self.vis.destroy_window()

    def create_recording_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = os.path.join(os.getcwd(), f"capture_{timestamp}")
        self.images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Init IMU CSV
        self.imu_csv_path = os.path.join(self.record_dir, "imu_data.csv")
        with open(self.imu_csv_path, "w") as f:
            f.write("frame_idx,timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
        
        # Init Trajectory File (Tum format: timestamp tx ty tz qx qy qz qw)
        self.traj_path = os.path.join(self.record_dir, "trajectory.txt")
        
        # Reset Pose Estimator on new recording?
        # Usually yes, so the recording starts at 0,0,0
        self.pose_estimator.reset()
        self.current_pose = np.identity(4)
        
        # Reset Map?
        # Optional: self.global_map.clear()
            
        print(f"Recording to: {self.record_dir}")
        self.frame_idx = 0

    def save_imu(self, accel, gyro, timestamp):
        if not self.recording or not self.record_dir:
            return
        
        with open(self.imu_csv_path, "a") as f:
            # simple format
            f.write(f"{self.frame_idx},{timestamp:.3f},{accel.x},{accel.y},{accel.z},{gyro.x},{gyro.y},{gyro.z}\n")
            
    def save_pose(self, timestamp):
        # Save current pose to trajectory file
        # Format: timestamp tx ty tz qx qy qz qw
        # Extract translation
        t = self.current_pose[:3, 3]
        # Extract rotation matrix and convert to quaternion
        # Open3D doesn't have direct Mat4 -> Quat in python easily available in all versions, 
        # so we use a small trick or helper if needed. 
        # Actually, SciPy is best for this, but trying to avoid extra deps if possible. 
        # Let's keep it simple: Save as 4x4 matrix in a separate file or row-major 16 floats.
        # Or: using CV2 Rodrigues? No, that's vector.
        # Let's save as flattened 4x4 matrix for now. Simpler to parse later.
        
        with open(self.traj_path, "a") as f:
            # Format: frame_idx, r00, r01, r02, tx, r10...
            flat_pose = self.current_pose.flatten()
            pose_str = ",".join([f"{x:.6f}" for x in flat_pose])
            f.write(f"{self.frame_idx},{timestamp},{pose_str}\n")

    def save_frame(self, color_image, depth_image):
        if not self.recording or not self.record_dir:
            return

        # Save images
        cv2.imwrite(os.path.join(self.images_dir, f"frame_{self.frame_idx:05d}.jpg"), color_image)
        # Save raw depth as PNG (16-bit)
        cv2.imwrite(os.path.join(self.images_dir, f"depth_{self.frame_idx:05d}.png"), depth_image)
        
        # Note: frame_idx increment is handled here, so IMU should be saved BEFORE or consistent with this.
        self.frame_idx += 1

    def save_point_cloud(self):
        """Saves the CURRENT frame's point cloud to the recording directory."""
        if self.last_points is None or not self.record_dir:
            print("No point cloud data to save.")
            return

        print("Saving Point Cloud...")
        # Convert to Open3D PointCloud
        # rs.pointcloud generates (N, 3) vertices
        verts = np.asanyarray(self.last_points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Colorize points
        if self.last_color_image is not None:
            # Flatten color image to match vertices (row-major)
            # OpenCV is BGR, Open3D needs RGB
            rgb_image = cv2.cvtColor(self.last_color_image, cv2.COLOR_BGR2RGB)
            colors = rgb_image.reshape(-1, 3) / 255.0 # Normalize to 0-1
        else:
            colors = None
        
        # Filter out zero points (invalid depth)
        valid_mask = (verts[:, 2] > 0) & (verts[:, 2] < 10) # Clip at 10 meters
        valid_verts = verts[valid_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_verts)
        
        if colors is not None:
            valid_colors = colors[valid_mask]
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            
        # Transform using current pose!
        pcd.transform(self.current_pose)
        
        # Save
        ply_path = os.path.join(self.record_dir, f"output_{self.frame_idx:05d}.ply") # Versioned PC
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Saved point cloud to {ply_path}")
        
    def update_global_map(self, color_image, depth_image):
        """Adds the current frame to the global map using current pose."""
        # 1. Generate Point Cloud for current frame (using Open3D for speed/ease vs RS wrapper)
        #    We use Open3D here to keep it compatible with the visualizer geometry
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, 
            depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.o3d_intrinsics
        )
        
        # 2. Transform to World
        pcd.transform(self.current_pose)
        
        # 3. Add to Global Map
        self.global_map += pcd
        
        # 4. Downsample to keep performance high (Voxel Grid Filter)
        #    0.05 = 5cm voxel size. Increase for performance, decrease for quality.
        self.global_map = self.global_map.voxel_down_sample(voxel_size=0.05)
        
        # 5. Update Visualizer
        self.vis.update_geometry(self.global_map)
        
        # 6. Optional: Move Visualizer Camera to follow? 
        #    For now, we let user control camera, but we could update view.

    def on_trackbar(self, val):
        self.decimation_percentage = val
        if self.decimation_percentage < 1:
            self.decimation_percentage = 1

    def run_gui(self):
        if not self.streaming:
            self.start()
            if not self.streaming:
                return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("PC Decimation %", self.window_name, 10, 100, self.on_trackbar)
        
        # Init Open3D Vis
        self.vis.create_window(window_name="3D Map (Live)", width=800, height=600, left=700, top=50)
        self.vis.add_geometry(self.global_map)
        # Add a coordinate frame
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(axis)
        self.map_initialized = True

        print("Controls:")
        print("  [SPACE] : Start/Stop Recording")
        print("  [P]     : Toggle Point Cloud Overlay")
        print("  [M]     : Toggle Live 3D Mapping")
        print("  [R]     : Reset Pose Tracking")
        print("  [Q]     : Quit")
        
        frame_count = 0

        while True:
            frames = self.pipeline.wait_for_frames()
            timestamp = frames.get_timestamp()
            frame_count += 1
            
            # Extract IMU Data
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            
            accel_data = accel_frame.as_motion_frame().get_motion_data() if accel_frame else None
            gyro_data = gyro_frame.as_motion_frame().get_motion_data() if gyro_frame else None

            if self.recording and accel_data and gyro_data:
                self.save_imu(accel_data, gyro_data, timestamp)

            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Update Pose Estimation
            if self.intrinsics_set:
                # Note: Open3D Odometry works best with grayscale or RGB.
                # We pass BGR, but we convert inside the update method if needed.
                # Wait, Open3D expects RGB usually.
                color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                new_pose = self.pose_estimator.update(color_rgb, depth_image, self.o3d_intrinsics)
                if new_pose is not None:
                    self.current_pose = new_pose
                    
            if self.recording:
                self.save_pose(timestamp)
            
            # Live Mapping Update
            # Only update every N frames to save CPU
            if self.mapping_enabled and frame_count % 10 == 0:
                self.update_global_map(color_image, depth_image)
            
            # Open3D Render Step
            self.vis.poll_events()
            self.vis.update_renderer()

            # Processing for Display
            # Colorize depth for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            display_image = color_image.copy()

            # Calculate Point Cloud if needed
            if self.show_pc_overlay or self.recording:
                self.pc.map_to(color_frame)
                points = self.pc.calculate(depth_frame)
                self.last_points = points
                self.last_color_image = color_image
                
                if self.show_pc_overlay:
                    # Overlay Logic:
                    # RealSense pointcloud object is just raw data. 
                    # Projecting it back to 2D is just... the depth map/color map.
                    # To visualize the "Point Cloud" specifically as "points", we can just sample pixels 
                    # and draw them on the screen to simulate a sparse cloud look.
                    
                    h, w = depth_image.shape
                    # Create a sparse mask based on decimation
                    mask = np.random.rand(h, w) < (self.decimation_percentage / 100.0)
                    # Highlight these pixels in the display image
                    # We can overlay the depth color on the RGB for these pixels
                    display_image[mask] = cv2.addWeighted(color_image[mask], 0.5, depth_colormap[mask], 0.5, 0).squeeze()

            # Status Text
            status_color = (0, 255, 0) if self.recording else (0, 0, 255)
            status_text = "RECORDING" if self.recording else "PAUSED"
            
            # Draw Translation info
            tx, ty, tz = self.current_pose[0, 3], self.current_pose[1, 3], self.current_pose[2, 3]
            pose_text = f"Pos: {tx:.2f}, {ty:.2f}, {tz:.2f}"
            map_text = f"Map: {'ON' if self.mapping_enabled else 'OFF'} ({len(self.global_map.points)} pts)"
            
            cv2.putText(display_image, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(display_image, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_image, map_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Combine views (Side by Side)
            # Resize depth colormap to match color image if needed (they are aligned so should be same)
            combined = np.hstack((display_image, depth_colormap))
            
            cv2.imshow(self.window_name, combined)

            # Handle Input
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord(' '): # Spacebar
                if self.recording:
                    # Stop recording
                    self.recording = False
                    # Save the accumulated Point Cloud (or just the last one for now)
                    self.save_point_cloud()
                    print("Recording stopped and data saved.")
                else:
                    # Start recording
                    self.create_recording_dir()
                    self.recording = True
                    print("Recording started...")
            elif key == ord('p'):
                self.show_pc_overlay = not self.show_pc_overlay
            elif key == ord('m'):
                self.mapping_enabled = not self.mapping_enabled
            elif key == ord('r'):
                self.pose_estimator.reset()
                self.current_pose = np.identity(4)
                self.global_map.clear() # Reset map too
                print("Pose and Map reset.")

            # Save frames if recording
            if self.recording:
                self.save_frame(color_image, depth_image)

        self.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealSenseRecorder()
    app.run_gui()