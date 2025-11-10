# 🚀 START HERE - Working Commands

## Google Drive Download Issue?

Don't worry! You have the **poster dataset already downloaded** and ready to use.

---

## ✅ Commands That Work Right Now

### Quick Test (2-5 minutes)
```bash
source venv/bin/activate
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 1000
```

### High Quality (30-60 minutes)
```bash
source venv/bin/activate
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 30000 --hq
```

### Compare 3DGS vs Instant-NGP (10-20 minutes)
```bash
source venv/bin/activate

# Train both models
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 5000
python examples/train_and_evaluate.py --dataset poster --model instant-ngp --iterations 5000

# Compare them
python examples/evaluation/compare_models.py \
    outputs/splatfacto_poster_i5000/poster/splatfacto/*/config.yml \
    outputs/instant-ngp_poster_i5000/poster/instant-ngp/*/config.yml \
    --names 3dgs ngp \
    --measure-speed \
    --output-dir results

# Visualize
python examples/visualizations/plot_metrics.py results/comparison_report.json
```

---

## 📊 What You'll Get

After running the quick test:
```json
{
  "psnr": "23-24 dB (good for 1000 iterations)",
  "ssim": "0.85-0.86 (good quality)",
  "fps": "90-100 fps (very fast!)",
  "lpips": "0.32 (perceptual quality)"
}
```

After running 30k iterations:
```json
{
  "psnr": "28-32 dB (excellent!)",
  "ssim": "0.92-0.96 (very good)",
  "fps": "80-100 fps (still fast!)",
  "training_time": "30-45 minutes"
}
```

---

## 🎬 Render a Video

After training, render a video:
```bash
# Find your model
CONFIG=$(find outputs/splatfacto_poster_i1000 -name "config.yml" -type f | head -1)

# Render video
ns-render interpolate --load-config $CONFIG --output-path poster_video.mp4 --frame-rate 30
```

---

## 📖 About the Poster Dataset

- **Type**: Real-world indoor scene
- **Resolution**: 1080x1920 pixels (high quality!)
- **Images**: 226 frames
- **Scene**: Indoor poster with good detail
- **Status**: ✓ Already downloaded and ready

This is perfect for testing the entire pipeline!

---

## 🔧 Want More Datasets?

See `DOWNLOAD_WORKAROUND.md` for:
- Manual Blender dataset download
- Alternative NeRFStudio datasets
- Workarounds for Google Drive rate limits

But **start with poster** - it works perfectly right now!

---

## ❓ What If I See Errors?

### "PyTorch 2.6 compatibility" warning
✓ Already fixed with patched eval script - ignore the FutureWarnings

### "Google Drive rate limit"
✓ Use poster dataset (already downloaded) or see `DOWNLOAD_WORKAROUND.md`

### "Out of memory"
✓ Training auto-downsamples to fit your GPU

### "Viewer crash"
✓ Normal - use `--viewer.quit-on-train-completion True` (already included in script)

---

## 🎯 Your First Command

Copy and paste this:

```bash
source venv/bin/activate
python examples/train_and_evaluate.py --dataset poster --model splatfacto --iterations 1000
```

Results in 2-5 minutes! 🚀

---

## 📁 Files to Read

1. **This file** (`START_HERE.md`) - You're reading it!
2. **`QUICK_COMMANDS.md`** - More command examples
3. **`DOWNLOAD_WORKAROUND.md`** - If you hit download issues
4. **`README_DATASET_SELECTION.md`** - Full guide to all datasets

---

## 🎉 Next Steps

After your first successful run:
1. Try high quality: add `--iterations 30000 --hq`
2. Compare models: train both 3DGS and Instant-NGP
3. Render videos: use `ns-render` commands above
4. Download more datasets: see `DOWNLOAD_WORKAROUND.md`

**But first - run the command above and see it work!**
