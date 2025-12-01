# 3DGS Studio - RealSense to Gaussian Splatting Pipeline

A complete GUI application to capture data from an Intel RealSense camera, process it using Structure-from-Motion (COLMAP), train a 3D Gaussian Splatting model (Nerfstudio), and export the result.

## Features

*   **RealSense Capture**: Record RGB and Depth data with optimized settings for motion (Global Shutter emulation).
*   **Smart Filtering**: Automatically detect and remove blurry frames to improve reconstruction quality.
*   **Integrated SfM**: Run COLMAP (via Nerfstudio wrappers) directly from the GUI to generate sparse point clouds and camera poses.
*   **3D Visualization**: Built-in viewer to inspect the sparse point cloud before training.
*   **Training**: Launch Nerfstudio 3DGS training with one click.
*   **Export**: auto-export trained models to `.ply` for viewing in external tools.

## Prerequisites

*   **Hardware**:
    *   Intel RealSense D400 Series Camera (D435i, D455, etc.)
    *   NVIDIA GPU (Required for Nerfstudio training)
*   **Software**:
    *   Linux (Ubuntu 22.04 recommended)
    *   COLMAP (must be installed and in system PATH)
    *   CUDA Toolkit (compatible with your PyTorch version)

## Installation

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
    pip install pyrealsense2 numpy opencv-python open3d pyqt5 pyvistaqt
    ```

## Usage

1.  **Start the GUI:**
    ```bash
    python test_application.py
    ```

2.  **Workflow:**

    *   **Step 0: Capture**
        *   Point camera at the object.
        *   Click **Start Recording**.
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

*   **Camera not detected:** Unplug and replug the RealSense camera. Ensure no other app is using it (like `realsense-viewer`).
*   **SfM Fails:** Ensure you have enough texture in the scene and overlap between frames. Avoid fast motion.
*   **Training GPU Error:** Ensure your PyTorch version matches your CUDA version (`nvcc --version`).