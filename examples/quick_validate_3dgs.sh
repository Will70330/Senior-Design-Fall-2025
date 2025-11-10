#!/bin/bash
# Quick validation script for 3DGS-MCMC pipeline

set -e  # Exit on error

echo "========================================="
echo "3DGS-MCMC Quick Validation"
echo "========================================="
echo ""

# Activate environment
source venv/bin/activate

# Check dataset
echo "1. Checking dataset..."
if [ ! -d "examples/datasets/nerfstudio/poster" ]; then
    echo "Dataset not found! Downloading..."
    ns-download-data nerfstudio --capture-name poster --save-dir examples/datasets
fi
echo "✓ Dataset ready ($(ls examples/datasets/nerfstudio/poster/images/ | wc -l) images at 1080x1920)"
echo ""

# Train 3DGS-MCMC
echo "2. Training 3DGS-MCMC (1000 iterations, ~2-5 minutes)..."
ns-train splatfacto \
    --data examples/datasets/nerfstudio/poster \
    --max-num-iterations 1000 \
    --viewer.quit-on-train-completion True \
    --output-dir outputs/quick_3dgs_validation

echo ""
echo "✓ Training complete!"
echo ""

# Find config
CONFIG=$(find outputs/quick_3dgs_validation -name "config.yml" -type f 2>/dev/null | head -1)

if [ -z "$CONFIG" ]; then
    echo "Error: Could not find config.yml in outputs/quick_3dgs_validation"
    echo "Directory structure:"
    ls -R outputs/quick_3dgs_validation
    exit 1
fi

echo "Found config: $CONFIG"

# Evaluate (using patched script for PyTorch 2.6 compatibility)
echo "3. Evaluating model..."
python examples/eval_model.py \
    --load-config $CONFIG \
    --output-path outputs/quick_3dgs_validation/eval_results.json

echo ""
echo "✓ Evaluation complete!"
echo ""

# Display results
echo "========================================="
echo "Results:"
echo "========================================="
if [ -f "outputs/quick_3dgs_validation/eval_results.json" ]; then
    python3 -c "
import json
with open('outputs/quick_3dgs_validation/eval_results.json', 'r') as f:
    data = json.load(f)
    print(f\"PSNR: {data.get('psnr', data.get('psnr_mean', 'N/A')):.2f} dB\")
    print(f\"SSIM: {data.get('ssim', data.get('ssim_mean', 'N/A')):.4f}\")
"
fi

echo ""
echo "Model saved to: outputs/quick_3dgs_validation/splatfacto/"
echo "Config: $CONFIG"
echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. View training visualization:"
echo "   ns-viewer --load-config $CONFIG"
echo ""
echo "2. Render video:"
echo "   ns-render interpolate --load-config $CONFIG --output-path render.mp4"
echo ""
echo "3. Compare with Instant-NGP:"
echo "   See examples/validation_3dgs_mcmc.md for full workflow"
echo ""
echo "✓ Validation complete!"
