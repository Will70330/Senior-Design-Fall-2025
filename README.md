# RealSense Data Recorder for 3D Gaussian Splatting

This tool captures RGB and Depth data from an Intel RealSense D455 camera, visualizes the stream, and saves point clouds and images for 3D reconstruction tasks.

## Prerequisites

- Intel RealSense D455 Camera (or compatible D400 series)
- Linux Environment (tested on Ubuntu)
- Python 3.8+
- `librealsense2` installed on the system (optional if using `pyrealsense2` wheels, but recommended for udev rules).

## Setup

1. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install pyrealsense2 numpy opencv-python open3d
   ```

3. **USB Permissions (Linux):**
   Ensure your user has access to the USB device or proper udev rules are installed. If you get a "No device detected" error, try unplugging/replugging or running with `sudo` (not recommended for permanent use).

## Usage

Run the recorder:
```bash
python realsense_recorder.py
```

### Controls

- **SPACE**: Start / Stop Recording.
  - **Start**: Creates a new folder `capture_YYYYMMDD_HHMMSS/`.
  - **Stop**: Saves the last generated Point Cloud to `output.ply` in that folder and stops saving images.
- **P**: Toggle "Point Cloud Overlay" mode.
  - Visualizes a sparse set of points (based on depth) overlaid on the RGB feed.
- **Slider (Top)**: Adjusts the percentage of points visualized in the overlay (1% - 100%).
- **Q** or **ESC**: Quit the application.

## Output Structure

When a recording is finished, a directory is created:

```
capture_20251130_120000/
├── output.ply          # The Point Cloud file (Open3D format)
├── imu_data.csv        # Accelerometer and Gyroscope data
└── images/             # Per-frame data
    ├── frame_00000.jpg # RGB Image
    ├── depth_00000.png # Raw 16-bit Depth Image
    ├── frame_00001.jpg
    └── ...
```
