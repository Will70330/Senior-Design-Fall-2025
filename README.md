# 3DGS Studio - RealSense to Gaussian Splatting Pipeline

A complete GUI application to capture data from an Intel RealSense camera, process it using Structure-from-Motion (COLMAP), train a 3D Gaussian Splatting model (Nerfstudio), and export the result.

## Features

*   **RealSense Capture**: Record RGB and Depth data with optimized settings for motion (Global Shutter emulation).
*   **Wireless Streaming**: Stream data wirelessly from a Raspberry Pi to your workstation for processing.
*   **Smart Filtering**: Automatically detect and remove blurry frames to improve reconstruction quality.
*   **Integrated SfM**: Run COLMAP (via Nerfstudio wrappers) directly from the GUI to generate sparse point clouds and camera poses.
*   **3D Visualization**: Built-in viewer to inspect the sparse point cloud before training.
*   **Training**: Launch Nerfstudio 3DGS training with one click.
*   **Export**: auto-export trained models to `.ply` for viewing in external tools.

## Prerequisites

*   **Hardware**:
    *   **Workstation**: NVIDIA GPU (Required for Nerfstudio training)
    *   **Camera Host (Optional)**: Raspberry Pi 4/5 with RealSense connected
    *   Intel RealSense D400 Series Camera (D435i, D455, etc.)
*   **Software**:
    *   Linux (Ubuntu 22.04 recommended)
    *   COLMAP (must be installed and in system PATH)
    *   CUDA Toolkit (compatible with your PyTorch version)

## Installation (Workstation)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Senior-Design-Fall-2025
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Nerfstudio:**
    Follow the [official Nerfstudio installation guide](https://docs.nerf.studio/quickstart/installation.html).
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install nerfstudio
    ```

4.  **Install App Dependencies:**
    ```bash
    pip install pyrealsense2 numpy opencv-python open3d pyqt5 pyvistaqt pyzmq
    ```

## Wireless Setup (Raspberry Pi)

To capture data wirelessly, run the server script on your Raspberry Pi.

1.  **Install Dependencies on Pi:**
    ```bash
    pip install pyrealsense2 numpy opencv-python pyzmq
    ```

2.  **Run the Server:**
    Transfer `wireless_comms.py` to the Pi and run it:
    ```bash
    python3 wireless_comms.py
    ```
    The script will automatically detect the camera and broadcast its presence on the network.

## Usage

1.  **Start the GUI (Workstation):**
    ```bash
    python test_application.py
    ```

2.  **Connect to Camera:**
    *   **Local USB**: The app defaults to local USB mode.
    *   **Wireless**: 
        1.  Click **⚙ Settings**.
        2.  Change **Connection Mode** to `Wireless (Network)`.
        3.  Click **Auto Detect** to find the Pi (or enter IP manually).
        4.  Click **OK**. The status bar should show **Wireless: Connected 🟢**.

3.  **Workflow:**

    *   **Step 0: Capture**
        *   Point camera at the object.
        *   Click **Start Recording**. (In wireless mode, images are streamed and saved to your workstation).
        *   Move slowly in an arc or circle around the object.
        *   Click **Stop Recording**.

    *   **Step 1: Process Images**
        *   Click **1. Process Images**.
        *   This filters out blurry frames based on Laplacian variance.
        *   *Tip:* If you lose too many images, click **Reset Images** to restore them.

    *   **Step 2: Generate Sparse PC (SfM)**
        *   Click **2. Generate Sparse PC**.
        *   This runs COLMAP to calculate camera poses.
        *   Wait for the process to finish. A sparse point cloud will appear in the viewer.
        *   *Note:* If the point cloud looks incoherent (points scattered everywhere), retry the capture with smoother motion.

    *   **Step 3: Train 3DGS**
        *   Click **3. Train 3DGS**.
        *   This launches `ns-train` in the background.
        *   Open the displayed localhost link (usually `http://localhost:7007`) to watch the training in real-time.

    *   **Step 4: Export**
        *   Click **4. Export PLY**.
        *   The application will automatically find the trained model configuration and export the final Gaussian Splat to `capture_.../gs_export/splat.ply`.

## Output Structure

Data is saved in `capture_TIMESTAMP` folders:

```
capture_20251201_120000/
├── images/              # Good frames used for training
├── rejected_images/     # Blurry frames (can be restored)
├── colmap/              # COLMAP database and sparse reconstruction
├── outputs/             # Nerfstudio training checkpoints
├── gs_export/           # Final exported .ply files
└── transforms.json      # Nerfstudio camera poses
```

## Troubleshooting

*   **Wireless Discovery Fails:** Ensure both devices are on the same WiFi network. Firewalls may block UDP broadcasts (Port 5554). You can manually enter the Pi's IP address in Settings.
*   **Camera not detected:** Unplug and replug the RealSense camera. Ensure no other app is using it (like `realsense-viewer`).
*   **SfM Fails:** Ensure you have enough texture in the scene and overlap between frames. Avoid fast motion.
*   **Training GPU Error:** Ensure your PyTorch version matches your CUDA version (`nvcc --version`).