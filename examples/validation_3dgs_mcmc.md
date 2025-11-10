# 3D Gaussian Splatting MCMC Validation Guide

## Quick Validation Commands

### 1. Setup
```bash
source venv/bin/activate
```

### 2. Verify Dataset (Full Resolution)
```bash
# Check dataset structure
ls -lh examples/datasets/nerfstudio/poster/

# The poster dataset has multiple resolutions:
# - images/      (1080x1920 - FULL RESOLUTION)
# - images_2/    (540x960  - 2x downsampled)
# - images_4/    (270x480  - 4x downsampled)
# - images_8/    (135x240  - 8x downsampled)

# Verify full resolution images
ls examples/datasets/nerfstudio/poster/images/ | wc -l
# Should show 226 images
```

### 3. Train 3DGS-MCMC at Full Resolution

**Quick Test (1000 iterations, ~5-10 minutes):**
```bash
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 1000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/test_3dgs_mcmc \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.continue-cull-post-densification False
```

**Full Training (30000 iterations, ~30-60 minutes, recommended):**
```bash
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/full_3dgs_mcmc \
    --pipeline.model.cull-alpha-thresh 0.005
```

**High Quality Training with MCMC-specific settings:**
```bash
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/hq_3dgs_mcmc \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.densify-grad-thresh 0.0002 \
    --pipeline.model.num-downscales 0
```

### 4. Train at Even Higher Resolution (4K Output)

If you want higher resolution OUTPUT (not just input), you can render at higher resolution:

```bash
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/4k_3dgs_mcmc \
    --pipeline.datamanager.camera-res-scale-factor 0.5 \
    --pipeline.model.cull-alpha-thresh 0.005
```

### 5. Evaluate the Trained Model

```bash
# Find your config file
CONFIG_PATH=$(find outputs/full_3dgs_mcmc/splatfacto -name "config.yml" -type f)

# Run evaluation
ns-eval \
    --load-config $CONFIG_PATH \
    --output-path outputs/full_3dgs_mcmc/eval_results.json
```

### 6. Render High-Resolution Output

**Render test images at full resolution:**
```bash
ns-render dataset \
    --load-config $CONFIG_PATH \
    --output-path outputs/full_3dgs_mcmc/renders/ \
    --rendered-output-names rgb
```

**Render a camera path video at higher resolution:**
```bash
ns-render interpolate \
    --load-config $CONFIG_PATH \
    --output-path outputs/full_3dgs_mcmc/video.mp4 \
    --rendered-output-names rgb \
    --frame-rate 30 \
    --output-format video \
    --interpolation-steps 60
```

### 7. Compare with Instant-NGP

```bash
# Train Instant-NGP for comparison
ns-train instant-ngp \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/full_instant_ngp

# Run comparison
python examples/evaluation/compare_models.py \
    outputs/full_instant_ngp/instant-ngp/*/config.yml \
    outputs/full_3dgs_mcmc/splatfacto/*/config.yml \
    --names instant-ngp 3dgs-mcmc \
    --measure-speed \
    --output-dir examples/evaluation/results

# View results
cat examples/evaluation/results/comparison_report.md
```

### 8. Visualize Results with Training Time

```bash
python examples/visualizations/plot_metrics.py \
    examples/evaluation/results/comparison_report.json \
    --output-dir examples/evaluation/results

# View the plots
ls -lh examples/evaluation/results/*.png
```

## Key 3DGS-MCMC Parameters

### Resolution Control
- `--pipeline.datamanager.camera-res-scale-factor`: Scale input images
  - 1.0 = full resolution (1080x1920 for poster dataset)
  - 0.5 = 2x higher resolution (2160x3840)
  - 2.0 = half resolution (540x960)

### Quality Settings
- `--pipeline.model.cull-alpha-thresh`: Lower = more Gaussians = higher quality
  - Default: 0.1
  - High quality: 0.005
  - Ultra quality: 0.001

- `--pipeline.model.densify-grad-thresh`: Gradient threshold for densification
  - Default: 0.0002
  - More detail: 0.0001

- `--pipeline.model.num-downscales`: Number of resolution levels
  - 0 = train at full resolution only
  - 2 = use pyramid with 3 levels (default)

### MCMC-Specific (gsplat 1.5.3)
The splatfacto model in NeRFStudio uses gsplat which has built-in MCMC support for Gaussian relocation. Key features:
- Automatic MCMC-based Gaussian relocation
- Dead Gaussian detection and relocation
- More efficient than heuristic splitting/cloning

## Expected Outputs

### Training Time Comparison (1000 iterations on poster dataset)
- Instant-NGP: ~3-5 minutes
- 3DGS (splatfacto): ~2-4 minutes (faster!)

### Quality Comparison (30000 iterations)
- Instant-NGP PSNR: ~27-30 dB
- 3DGS PSNR: ~28-32 dB (often better)

### Rendering Speed
- Instant-NGP FPS: ~5-15 fps
- 3DGS FPS: ~30-100+ fps (much faster!)

## Troubleshooting

### Out of Memory
If you get CUDA out of memory errors:
```bash
# Reduce resolution
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --pipeline.datamanager.camera-res-scale-factor 2.0
```

### Slow Training
```bash
# Reduce number of downscales
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --pipeline.model.num-downscales 0
```

### Low Quality Results
```bash
# Increase iterations and adjust culling threshold
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 50000 \
    --pipeline.model.cull-alpha-thresh 0.001
```

## Complete Quick Validation Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train 3DGS-MCMC (quick test)
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 1000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/quick_3dgs

# 3. Evaluate
CONFIG=$(find outputs/quick_3dgs/splatfacto -name "config.yml" -type f)
ns-eval --load-config $CONFIG --output-path outputs/quick_3dgs/eval.json

# 4. Check results
cat outputs/quick_3dgs/eval.json

# 5. Optional: Compare with NeRF
ns-train instant-ngp \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 1000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/quick_ngp

python examples/evaluation/compare_models.py \
    outputs/quick_ngp/instant-ngp/*/config.yml \
    outputs/quick_3dgs/splatfacto/*/config.yml \
    --names ngp 3dgs \
    --measure-speed \
    --output-dir results

# 6. Visualize with training time included
python examples/visualizations/plot_metrics.py results/comparison_report.json
```

## Notes on Resolution

The poster dataset's **full resolution is already quite high** (1080x1920 pixels). This is sufficient for high-quality reconstruction. To get even higher resolution output:

1. **Use camera-res-scale-factor < 1.0**: This upsamples the training images
2. **Render at higher resolution**: Use the `--image-format png` and scale options in ns-render
3. **Train on higher resolution input**: If you have your own data, capture at 4K or higher

The default full resolution (1080x1920) should produce **very visible and high-quality results** with 3DGS-MCMC!
