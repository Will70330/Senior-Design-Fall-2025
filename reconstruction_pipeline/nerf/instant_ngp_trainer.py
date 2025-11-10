"""
Instant-NGP NeRF Training using NeRFStudio
Wrapper for training instant-ngp models with NeRFStudio
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstantNGPTrainer:
    """Train Instant-NGP NeRFs using NeRFStudio"""

    def __init__(self, nerfstudio_command: str = "ns-train"):
        """
        Args:
            nerfstudio_command: Command for NeRFStudio training
        """
        self.nerfstudio_command = nerfstudio_command

    def train(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        max_num_iterations: int = 30000,
        viewer_enabled: bool = True,
        viewer_port: int = 7007,
        experiment_name: Optional[str] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Train instant-ngp model

        Args:
            data_path: Path to dataset (NeRFStudio format)
            output_dir: Output directory for checkpoints
            max_num_iterations: Maximum training iterations
            viewer_enabled: Enable web viewer
            viewer_port: Port for web viewer
            experiment_name: Name for experiment
            additional_args: Additional command-line arguments

        Returns:
            Return code from training process
        """
        logger.info("=" * 60)
        logger.info("Starting Instant-NGP Training with NeRFStudio")
        logger.info("=" * 60)

        # Build command
        cmd = [
            self.nerfstudio_command,
            "instant-ngp",
            "--data", str(data_path),
            "--max-num-iterations", str(max_num_iterations),
        ]

        # Add output directory
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])

        # Add experiment name
        if experiment_name:
            cmd.extend(["--experiment-name", experiment_name])

        # Viewer settings
        if not viewer_enabled:
            cmd.append("--viewer.quit-on-train-completion")
        else:
            cmd.extend([
                "--vis", "viewer",
                "--viewer.websocket-port", str(viewer_port)
            ])

        # Add additional arguments
        if additional_args:
            for key, value in additional_args.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("=" * 60)

        # Run training
        try:
            result = subprocess.run(
                cmd,
                check=False,
                text=True
            )
            return result.returncode

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            return 1

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1

    def export_model(
        self,
        checkpoint_path: str,
        output_path: str,
        output_format: str = "ply"
    ) -> int:
        """
        Export trained model

        Args:
            checkpoint_path: Path to model checkpoint
            output_path: Output path
            output_format: Output format (ply, obj, etc.)

        Returns:
            Return code
        """
        logger.info(f"Exporting model to {output_format} format...")

        cmd = [
            "ns-export",
            output_format,
            "--load-config", checkpoint_path,
            "--output-dir", str(output_path)
        ]

        result = subprocess.run(cmd, check=False)
        return result.returncode

    def render_video(
        self,
        checkpoint_path: str,
        output_path: str,
        camera_path_filename: str = "camera_path.json",
        rendered_output_names: str = "rgb"
    ) -> int:
        """
        Render video from trained model

        Args:
            checkpoint_path: Path to model checkpoint
            output_path: Output video path
            camera_path_filename: Camera path file
            rendered_output_names: Output types to render

        Returns:
            Return code
        """
        logger.info("Rendering video...")

        cmd = [
            "ns-render",
            "camera-path",
            "--load-config", checkpoint_path,
            "--camera-path-filename", camera_path_filename,
            "--output-path", str(output_path),
            "--rendered-output-names", rendered_output_names
        ]

        result = subprocess.run(cmd, check=False)
        return result.returncode


def train_instant_ngp(
    data_path: str,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    max_iterations: int = 30000,
    enable_viewer: bool = True,
    viewer_port: int = 7007
) -> int:
    """
    Convenience function to train instant-ngp

    Args:
        data_path: Path to dataset
        output_dir: Output directory
        experiment_name: Experiment name
        max_iterations: Max training iterations
        enable_viewer: Enable web viewer
        viewer_port: Viewer port

    Returns:
        Return code
    """
    trainer = InstantNGPTrainer()

    return trainer.train(
        data_path=data_path,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_num_iterations=max_iterations,
        viewer_enabled=enable_viewer,
        viewer_port=viewer_port
    )
