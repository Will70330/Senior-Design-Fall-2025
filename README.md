# Senior Design - Novel View Synthesis Pipeline

Team #11's Senior Design Project - Drone-based scene/object reconstruction with comparative analysis of novel view synthesis methods.

## Project Overview

Custom software pipeline comparing two state-of-the-art novel view synthesis methods:
- **3D Gaussian Splatting with MCMC** (via gsplat)
- **Neural Radiance Fields** (Instant-NGP via NeRFStudio)

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (requires CUDA for GPU acceleration)
pip install -r requirements.txt
```

### Basic Usage Pipeline

**1. Extract frames from video:**
```bash
python scripts/extract_frames.py your_video.mp4
# Output: data/extracted_frames/your_video/
```

**2. Run Structure from Motion (COLMAP):**
```bash
python scripts/run_colmap.py data/extracted_frames/your_video
# Output: data/sfm_output/your_video/
```

**3. Convert to NeRFStudio format:**
```bash
ns-process-data images --data data/extracted_frames/your_video \
    --output-dir data/processed/your_video
```

**4. Train NeRF (Instant-NGP):**
```bash
python scripts/train_nerf.py data/processed/your_video
# Web viewer: http://localhost:7007
```

**5. Train 3D Gaussian Splatting with MCMC:**
```bash
ns-train splatfacto --data data/processed/your_video
# Built-in viewer launches automatically
```

## Project Structure

```
reconstruction_pipeline/
├── frame_extraction/      # Video → optimal frames selection
│   └── video_processor.py
├── sfm/                   # COLMAP Structure from Motion
│   └── colmap_runner.py
├── nerf/                  # Instant-NGP NeRF training
│   └── instant_ngp_trainer.py
├── gaussian_splatting/    # 3DGS-MCMC (TBD)
├── dataloaders/           # NeRFStudio dataset loaders
│   └── nerfstudio_dataloader.py
└── visualization/         # Viser 3D visualization (TBD)

data/
├── raw_videos/           # Input videos
├── extracted_frames/     # Extracted frames
├── sfm_output/          # COLMAP sparse reconstructions
└── models/              # Trained models

scripts/
├── extract_frames.py    # Frame extraction CLI
├── run_colmap.py        # COLMAP pipeline CLI
└── train_nerf.py        # NeRF training CLI

context/
├── pipeline_architecture.md   # Technical architecture details
└── visualization_options.md   # Visualization library comparison

configs/                 # Configuration files (TBD)
Documentation/           # Project documentation
```

## Pipeline Stages

### 1. Frame Extraction
Automatically selects optimal frames from high-frame-rate video based on:
- Blur detection (Laplacian variance)
- Frame similarity (avoiding redundant frames)
- Configurable sampling rate

**Usage:**
```bash
python scripts/extract_frames.py video.mp4 \
    --blur-threshold 100.0 \
    --similarity-threshold 0.95 \
    --min-interval 5
```

### 2. Structure from Motion (SfM)
Uses COLMAP to generate:
- Sparse 3D point cloud
- Camera poses and intrinsics
- Feature matches

**Usage:**
```bash
python scripts/run_colmap.py data/extracted_frames/scene \
    --camera-model OPENCV \
    --matching-mode exhaustive
```

### 3. Novel View Synthesis

#### Option A: Instant-NGP (NeRF)
Fast neural radiance field training using instant-ngp:

```bash
# Train
python scripts/train_nerf.py data/processed/scene \
    --max-iterations 30000

# Export point cloud
ns-export pointcloud --load-config outputs/.../config.yml \
    --output-dir exports/pointcloud.ply

# Render video
ns-render camera-path --load-config outputs/.../config.yml \
    --camera-path-filename camera_path.json \
    --output-path renders/video.mp4
```

#### Option B: 3D Gaussian Splatting with MCMC
Using gsplat's MCMC-based optimization:

```bash
# Train with MCMC densification
ns-train splatfacto --data data/processed/scene \
    --pipeline.model.use-mcmc True

# Or use 3DGUT with MCMC for complex camera models
ns-train splatfacto --data data/processed/scene \
    --pipeline.model.use-3dgut True
```

## Key Technologies

- **NeRFStudio** (v1.0+): End-to-end NeRF framework with instant-ngp
- **gsplat** (v1.0+): CUDA-accelerated 3D Gaussian Splatting with MCMC
- **COLMAP** (v3.13+): Structure from Motion
- **PyTorch** (v2.0+): Deep learning framework
- **Viser** (v0.1+): Web-based 3D visualization
- **CUDA 12.x**: GPU acceleration (RTX 4090 recommended)

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- 16GB+ RAM
- 50GB+ storage for datasets and models

### Software
- Python 3.9+
- CUDA 12.x
- COLMAP (install separately: https://colmap.github.io/)

## Advanced Usage

### Custom Frame Extraction
```python
from reconstruction_pipeline.frame_extraction.video_processor import FrameExtractor

extractor = FrameExtractor(
    blur_threshold=100.0,
    similarity_threshold=0.95,
    min_frame_interval=5,
    max_frames=300
)
frame_paths = extractor.extract_frames("video.mp4", "output/frames")
```

### Custom COLMAP Pipeline
```python
from reconstruction_pipeline.sfm.colmap_runner import COLMAPRunner

runner = COLMAPRunner()
outputs = runner.run_full_pipeline(
    image_path="frames/",
    output_path="sfm_output/",
    camera_model="OPENCV",
    matching_mode="exhaustive"
)
```

### NeRFStudio Dataloader
```python
from reconstruction_pipeline.dataloaders.nerfstudio_dataloader import (
    create_nerfstudio_dataloader
)

dataloader = create_nerfstudio_dataloader(
    data_dir="data/processed/scene",
    batch_size=4,
    scale_factor=0.5
)

for batch in dataloader:
    images = batch["image"]
    poses = batch["transform_matrix"]
    # ... training code
```

## Visualization

Web-based viewers automatically launch during training:
- **NeRF**: http://localhost:7007
- **3DGS**: http://localhost:7007

For custom visualizations, we recommend **Viser** (see `context/visualization_options.md`).

## Development Status

- ✅ Frame extraction module
- ✅ COLMAP SfM integration
- ✅ NeRFStudio dataloader
- ✅ Instant-NGP training wrapper
- ⏳ 3DGS-MCMC training wrapper
- ⏳ Viser visualization integration
- ⏳ Comparison metrics and analysis tools

## Documentation

- `context/pipeline_architecture.md` - Technical architecture and design decisions
- `context/visualization_options.md` - Visualization library comparison
- `CLAUDE.md` - Development guide for Claude Code

## Project Links

- Design Document: https://docs.google.com/document/d/1CyGZwGUgogyvw26dscT4mX_WAf277EYH/edit

## License

[TBD]

## Acknowledgments

- NeRFStudio team for instant-ngp and gsplat
- COLMAP developers
- 3D Gaussian Splatting research community
