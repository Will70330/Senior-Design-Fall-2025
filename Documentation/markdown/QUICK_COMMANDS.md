# Quick Command Reference

## Activate Environment (Always run first!)
```bash
source venv/bin/activate
```

---

## Single Dataset Examples

### Train 3DGS on Lego (Quick Test - 2-5 min)
```bash
python examples/train_and_evaluate.py --dataset lego --model splatfacto --iterations 1000
```

### Train 3DGS on Poster (Already Downloaded)
```bash
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 1000
```

### Train Instant-NGP on Chair
```bash
python examples/train_and_evaluate.py --dataset chair --model instant-ngp --iterations 1000
```

### High Quality 3DGS on Vegetation (30-60 min)
```bash
python examples/train_and_evaluate.py --dataset vegetation --model splatfacto --iterations 30000 --hq
```

---

## Multiple Dataset Examples

### Test Multiple Blender Datasets
```bash
python examples/train_and_evaluate.py --dataset lego chair drums --model splatfacto --iterations 1000
```

### Test Multiple Real-World Datasets
```bash
python examples/train_and_evaluate.py --dataset poster library kitchen --model splatfacto --iterations 5000
```

### Mix Both Types
```bash
python examples/train_and_evaluate.py --dataset lego poster vegetation --model splatfacto --iterations 1000
```

---

## Compare Different Models

### 3DGS vs Instant-NGP on Same Dataset
```bash
# Train both models
python examples/train_and_evaluate.py --dataset lego --model splatfacto --iterations 5000
python examples/train_and_evaluate.py --dataset lego --model instant-ngp --iterations 5000

# Compare them
python examples/evaluation/compare_models.py \
    outputs/splatfacto_lego_i5000/*/splatfacto/*/config.yml \
    outputs/instant-ngp_lego_i5000/*/instant-ngp/*/config.yml \
    --names 3dgs-lego ngp-lego \
    --measure-speed

# Visualize
python examples/visualizations/plot_metrics.py results/comparison_report.json
```

---

## Compare Same Model Across Datasets

```bash
# Train on multiple datasets
python examples/train_and_evaluate.py --dataset lego poster chair --model splatfacto --iterations 5000

# Compare results
python examples/evaluation/compare_models.py \
    outputs/splatfacto_lego_i5000/*/splatfacto/*/config.yml \
    outputs/splatfacto_poster_i5000/*/splatfacto/*/config.yml \
    outputs/splatfacto_chair_i5000/*/splatfacto/*/config.yml \
    --names lego poster chair \
    --measure-speed

# Visualize
python examples/visualizations/plot_metrics.py results/comparison_report.json
```

---

## Render Videos

### Render from Latest Model
```bash
# Find your config
CONFIG=$(find outputs -name "config.yml" -type f | head -1)

# Render interpolated camera path
ns-render interpolate \
    --load-config $CONFIG \
    --output-path video.mp4 \
    --frame-rate 30
```

### Render Test Views
```bash
ns-render dataset \
    --load-config $CONFIG \
    --output-path renders/ \
    --rendered-output-names rgb
```

---

## Available Datasets

### NeRFStudio (17 real-world datasets)
poster, bww_entrance, storefront, vegetation, library, campanile, desolation, redwoods2, Egypt, person, kitchen, plane, dozer, floating-tree, aspen, stump, sculpture

### Blender (8 synthetic datasets)
lego, chair, drums, ficus, hotdog, materials, mic, ship

---

## Quick Troubleshooting

### "Out of memory"
```bash
# Use downsampled images
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 1000
# (Training automatically uses reasonable resolution)
```

### "Dataset not found"
```bash
# Script auto-downloads, but if it fails, download manually:
ns-download-data nerfstudio --capture-name lego --save-dir examples/datasets
# or
ns-download-data blender --save-dir examples/datasets
```

### "Want to see all options"
```bash
python examples/train_and_evaluate.py --help
```

---

## Model Types

- **splatfacto** (default): 3D Gaussian Splatting with MCMC
  - Fastest rendering (80-100+ FPS)
  - Good quality
  - Best for real-time applications

- **instant-ngp**: Instant Neural Graphics Primitives
  - Good quality
  - Moderate rendering speed (10-20 FPS)
  - Smaller model size

- **nerfacto**: Standard NeRF implementation
  - High quality
  - Slower rendering (5-15 FPS)
  - Good for static scenes

---

## Iteration Guidelines

- **1000**: Quick test (2-5 minutes)
- **5000**: Good quality (10-20 minutes)
- **10000**: High quality (20-40 minutes)
- **30000**: Production quality (40-90 minutes)
- **50000**: Maximum quality (90-150 minutes)

Use `--hq` flag for even better quality with splatfacto!

---

## Output Structure

Results saved to: `outputs/<model>_<dataset>_i<iterations>/`

Each run contains:
- `config.yml` - Model configuration
- `nerfstudio_models/*.ckpt` - Model checkpoints
- `eval_results.json` - Evaluation metrics
