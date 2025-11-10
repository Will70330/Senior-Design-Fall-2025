"""
Metrics for evaluating novel view synthesis quality
Includes PSNR, SSIM, LPIPS, and rendering performance metrics
"""

import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class MetricsTracker:
    """Track and compute metrics for novel view synthesis"""

    def __init__(self):
        self.metrics_history = []
        self.timing_history = []

    def compute_psnr(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        data_range: Optional[float] = None
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio

        Args:
            pred: Predicted image (H, W, C) in range [0, 1] or [0, 255]
            target: Target image (H, W, C) in same range as pred
            data_range: Maximum possible pixel value (auto-detected if None)

        Returns:
            PSNR value in dB
        """
        if not SKIMAGE_AVAILABLE:
            # Fallback implementation
            mse = np.mean((pred - target) ** 2)
            if mse == 0:
                return float('inf')

            if data_range is None:
                data_range = pred.max() - pred.min()

            return 20 * np.log10(data_range / np.sqrt(mse))

        if data_range is None:
            data_range = pred.max() - pred.min()

        return peak_signal_noise_ratio(target, pred, data_range=data_range)

    def compute_ssim(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        data_range: Optional[float] = None,
        multichannel: bool = True
    ) -> float:
        """
        Compute Structural Similarity Index

        Args:
            pred: Predicted image
            target: Target image
            data_range: Maximum possible pixel value
            multichannel: Whether to compute across channels

        Returns:
            SSIM value in range [0, 1]
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for SSIM")

        if data_range is None:
            data_range = pred.max() - pred.min()

        return structural_similarity(
            target,
            pred,
            data_range=data_range,
            channel_axis=2 if multichannel and pred.ndim == 3 else None
        )

    def compute_mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Squared Error"""
        return float(np.mean((pred - target) ** 2))

    def compute_mae(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Absolute Error"""
        return float(np.mean(np.abs(pred - target)))

    @torch.no_grad()
    def measure_inference_time(
        self,
        model_fn,
        inputs,
        num_warmup: int = 5,
        num_iterations: int = 100,
        use_cuda: bool = True
    ) -> Dict[str, float]:
        """
        Measure inference time and FPS

        Args:
            model_fn: Callable that takes inputs and returns outputs
            inputs: Input to the model
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            use_cuda: Whether to use CUDA events for timing

        Returns:
            Dictionary with timing statistics
        """
        if use_cuda and torch.cuda.is_available():
            # Use CUDA events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Warmup
            for _ in range(num_warmup):
                _ = model_fn(inputs)

            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(num_iterations):
                start_event.record()
                _ = model_fn(inputs)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

            times = np.array(times)

        else:
            # CPU timing
            for _ in range(num_warmup):
                _ = model_fn(inputs)

            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model_fn(inputs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            times = np.array(times)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "fps": 1000.0 / float(np.mean(times)),
        }

    def evaluate_images(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        compute_ssim: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a set of predicted images

        Args:
            predictions: List of predicted images
            targets: List of target images
            compute_ssim: Whether to compute SSIM (slower)

        Returns:
            Dictionary of averaged metrics
        """
        assert len(predictions) == len(targets), "Mismatch in number of images"

        psnr_values = []
        ssim_values = []
        mse_values = []
        mae_values = []

        for pred, target in zip(predictions, targets):
            # Normalize to [0, 1] if needed
            if pred.max() > 1.0:
                pred = pred / 255.0
            if target.max() > 1.0:
                target = target / 255.0

            psnr_values.append(self.compute_psnr(pred, target, data_range=1.0))
            mse_values.append(self.compute_mse(pred, target))
            mae_values.append(self.compute_mae(pred, target))

            if compute_ssim and SKIMAGE_AVAILABLE:
                ssim_values.append(self.compute_ssim(pred, target, data_range=1.0))

        metrics = {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "mse_mean": float(np.mean(mse_values)),
            "mae_mean": float(np.mean(mae_values)),
        }

        if ssim_values:
            metrics["ssim_mean"] = float(np.mean(ssim_values))
            metrics["ssim_std"] = float(np.std(ssim_values))

        return metrics

    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics to history"""
        entry = {"metrics": metrics}
        if step is not None:
            entry["step"] = step
        entry["timestamp"] = time.time()
        self.metrics_history.append(entry)

    def save_metrics(self, filepath: Path):
        """Save metrics history to JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump({
                "metrics_history": self.metrics_history,
                "timing_history": self.timing_history
            }, f, indent=2)

    @classmethod
    def load_metrics(cls, filepath: Path) -> 'MetricsTracker':
        """Load metrics history from JSON"""
        tracker = cls()

        with open(filepath, "r") as f:
            data = json.load(f)

        tracker.metrics_history = data.get("metrics_history", [])
        tracker.timing_history = data.get("timing_history", [])

        return tracker


def compute_batch_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for a batch of images (PyTorch tensors)

    Args:
        predictions: Tensor of shape (B, C, H, W) or (B, H, W, C)
        targets: Tensor of same shape as predictions

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Ensure (B, H, W, C) format
    if predictions.shape[1] == 3 or predictions.shape[1] == 1:
        predictions = np.transpose(predictions, (0, 2, 3, 1))
        targets = np.transpose(targets, (0, 2, 3, 1))

    # Compute metrics
    tracker = MetricsTracker()
    pred_list = [predictions[i] for i in range(predictions.shape[0])]
    target_list = [targets[i] for i in range(targets.shape[0])]

    return tracker.evaluate_images(pred_list, target_list)
