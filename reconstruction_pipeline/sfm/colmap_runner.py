"""
COLMAP Structure from Motion Pipeline
Generates sparse 3D point cloud and camera poses from image frames
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COLMAPRunner:
    """Run COLMAP Structure from Motion pipeline"""

    def __init__(self, colmap_executable: str = "colmap"):
        """
        Args:
            colmap_executable: Path to COLMAP executable
        """
        self.colmap_executable = colmap_executable

        # Verify COLMAP is installed
        if not self._check_colmap_installed():
            raise RuntimeError(
                "COLMAP not found. Please install COLMAP: "
                "https://colmap.github.io/"
            )

    def _check_colmap_installed(self) -> bool:
        """Check if COLMAP is installed"""
        return shutil.which(self.colmap_executable) is not None

    def _run_command(self, args: list) -> int:
        """Run COLMAP command"""
        cmd = [self.colmap_executable] + args
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"COLMAP command failed: {result.stderr}")
            raise RuntimeError(f"COLMAP failed with return code {result.returncode}")

        return result.returncode

    def extract_features(
        self,
        image_path: str,
        database_path: str,
        camera_model: str = "OPENCV",
        single_camera: bool = False
    ) -> None:
        """
        Extract features from images

        Args:
            image_path: Path to directory containing images
            database_path: Path to COLMAP database file
            camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
            single_camera: Whether to use single camera for all images
        """
        logger.info("Extracting features...")

        args = [
            "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--ImageReader.camera_model", camera_model,
        ]

        if single_camera:
            args.extend(["--ImageReader.single_camera", "1"])

        self._run_command(args)
        logger.info("Feature extraction completed")

    def match_features(
        self,
        database_path: str,
        matching_mode: str = "exhaustive"
    ) -> None:
        """
        Match features between images

        Args:
            database_path: Path to COLMAP database
            matching_mode: Matching mode (exhaustive, sequential, vocab_tree, spatial)
        """
        logger.info(f"Matching features using {matching_mode} matching...")

        if matching_mode == "exhaustive":
            args = [
                "exhaustive_matcher",
                "--database_path", database_path,
            ]
        elif matching_mode == "sequential":
            args = [
                "sequential_matcher",
                "--database_path", database_path,
            ]
        elif matching_mode == "vocab_tree":
            args = [
                "vocab_tree_matcher",
                "--database_path", database_path,
            ]
        elif matching_mode == "spatial":
            args = [
                "spatial_matcher",
                "--database_path", database_path,
            ]
        else:
            raise ValueError(f"Unknown matching mode: {matching_mode}")

        self._run_command(args)
        logger.info("Feature matching completed")

    def run_mapper(
        self,
        database_path: str,
        image_path: str,
        output_path: str
    ) -> None:
        """
        Run sparse reconstruction (mapper)

        Args:
            database_path: Path to COLMAP database
            image_path: Path to images directory
            output_path: Path to output sparse reconstruction directory
        """
        logger.info("Running mapper (sparse reconstruction)...")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        args = [
            "mapper",
            "--database_path", database_path,
            "--image_path", image_path,
            "--output_path", str(output_path),
        ]

        self._run_command(args)
        logger.info(f"Mapper completed. Output: {output_path}")

    def convert_model(
        self,
        input_path: str,
        output_path: str,
        output_type: str = "TXT"
    ) -> None:
        """
        Convert COLMAP model to different format

        Args:
            input_path: Path to input model directory
            output_path: Path to output directory
            output_type: Output format (TXT, BIN, NVM, Bundler, VRML, PLY, etc.)
        """
        logger.info(f"Converting model to {output_type} format...")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        args = [
            "model_converter",
            "--input_path", input_path,
            "--output_path", str(output_path),
            "--output_type", output_type,
        ]

        self._run_command(args)
        logger.info(f"Model converted: {output_path}")

    def run_full_pipeline(
        self,
        image_path: str,
        output_path: str,
        camera_model: str = "OPENCV",
        matching_mode: str = "exhaustive",
        single_camera: bool = False
    ) -> Dict[str, str]:
        """
        Run complete COLMAP SfM pipeline

        Args:
            image_path: Path to directory containing images
            output_path: Path to output directory
            camera_model: Camera model
            matching_mode: Feature matching mode
            single_camera: Use single camera for all images

        Returns:
            Dictionary with paths to outputs
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create database path
        database_path = output_path / "database.db"

        # Create sparse output directory
        sparse_path = output_path / "sparse"
        sparse_path.mkdir(exist_ok=True)

        logger.info("=" * 60)
        logger.info("Starting COLMAP SfM Pipeline")
        logger.info("=" * 60)
        logger.info(f"Image path: {image_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Camera model: {camera_model}")
        logger.info(f"Matching mode: {matching_mode}")
        logger.info("=" * 60)

        # Step 1: Feature extraction
        self.extract_features(
            image_path=str(image_path),
            database_path=str(database_path),
            camera_model=camera_model,
            single_camera=single_camera
        )

        # Step 2: Feature matching
        self.match_features(
            database_path=str(database_path),
            matching_mode=matching_mode
        )

        # Step 3: Sparse reconstruction
        self.run_mapper(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(sparse_path)
        )

        # Step 4: Convert to text format for easy reading
        # Find the reconstruction directory (usually "0")
        reconstruction_dirs = list(sparse_path.glob("*"))
        if reconstruction_dirs:
            latest_recon = sorted(reconstruction_dirs)[-1]
            txt_output = output_path / "sparse_txt"

            self.convert_model(
                input_path=str(latest_recon),
                output_path=str(txt_output),
                output_type="TXT"
            )
        else:
            logger.warning("No reconstruction found!")
            txt_output = None

        logger.info("=" * 60)
        logger.info("COLMAP Pipeline Completed!")
        logger.info("=" * 60)

        return {
            "database": str(database_path),
            "sparse": str(sparse_path),
            "sparse_txt": str(txt_output) if txt_output else None,
            "output": str(output_path)
        }


def run_colmap_sfm(
    image_path: str,
    output_path: str,
    camera_model: str = "OPENCV",
    matching_mode: str = "exhaustive",
    single_camera: bool = False,
    colmap_executable: str = "colmap"
) -> Dict[str, str]:
    """
    Convenience function to run COLMAP SfM pipeline

    Args:
        image_path: Path to images
        output_path: Output directory
        camera_model: COLMAP camera model
        matching_mode: Feature matching strategy
        single_camera: Use single camera assumption
        colmap_executable: Path to COLMAP binary

    Returns:
        Dictionary with output paths
    """
    runner = COLMAPRunner(colmap_executable=colmap_executable)
    return runner.run_full_pipeline(
        image_path=image_path,
        output_path=output_path,
        camera_model=camera_model,
        matching_mode=matching_mode,
        single_camera=single_camera
    )
