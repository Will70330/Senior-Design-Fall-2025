"""
NeRFStudio Dataset Dataloader
Loads data in NeRFStudio format for training and validation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NerfstudioDataset(Dataset):
    """
    Dataset class for NeRFStudio format data

    NeRFStudio format structure:
    data_dir/
        transforms.json     # Camera poses and intrinsics
        images/            # Training images
            frame_00000.jpg
            frame_00001.jpg
            ...
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        scale_factor: float = 1.0,
        load_images: bool = True
    ):
        """
        Args:
            data_dir: Path to dataset directory containing transforms.json
            split: Dataset split ("train", "val", "test")
            scale_factor: Scale factor for images (1.0 = original size)
            load_images: Whether to load images into memory
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.data_dir = Path(data_dir)
        self.split = split
        self.scale_factor = scale_factor
        self.load_images = load_images

        # Load transforms file
        transforms_file = self.data_dir / "transforms.json"
        if not transforms_file.exists():
            raise FileNotFoundError(f"transforms.json not found in {data_dir}")

        with open(transforms_file, "r") as f:
            self.meta = json.load(f)

        # Parse camera intrinsics
        self.parse_camera_intrinsics()

        # Parse frames
        self.frames = []
        for frame in self.meta.get("frames", []):
            frame_path = self.data_dir / frame["file_path"]

            # Handle different path formats
            if not frame_path.exists():
                # Try with .jpg extension
                frame_path = self.data_dir / (frame["file_path"] + ".jpg")
            if not frame_path.exists():
                # Try with .png extension
                frame_path = self.data_dir / (frame["file_path"] + ".png")

            if frame_path.exists():
                self.frames.append({
                    "file_path": frame_path,
                    "transform_matrix": np.array(frame["transform_matrix"], dtype=np.float32)
                })
            else:
                logger.warning(f"Image not found: {frame['file_path']}")

        if len(self.frames) == 0:
            raise ValueError(f"No valid frames found in {data_dir}")

        logger.info(f"Loaded {len(self.frames)} frames from {data_dir}")

        # Preload images if requested
        if self.load_images and CV2_AVAILABLE:
            logger.info("Preloading images...")
            self.images = []
            for frame in self.frames:
                img = cv2.imread(str(frame["file_path"]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.scale_factor != 1.0:
                    new_size = (
                        int(img.shape[1] * self.scale_factor),
                        int(img.shape[0] * self.scale_factor)
                    )
                    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

                self.images.append(img)
        else:
            self.images = None

    def parse_camera_intrinsics(self):
        """Parse camera intrinsics from metadata"""
        # Camera angle (field of view)
        if "camera_angle_x" in self.meta:
            self.camera_angle_x = self.meta["camera_angle_x"]
        else:
            self.camera_angle_x = None

        # Image dimensions
        if "w" in self.meta and "h" in self.meta:
            self.image_width = int(self.meta["w"] * self.scale_factor)
            self.image_height = int(self.meta["h"] * self.scale_factor)
        else:
            # Will be set from first image
            self.image_width = None
            self.image_height = None

        # Focal length
        if "fl_x" in self.meta and "fl_y" in self.meta:
            self.fl_x = self.meta["fl_x"] * self.scale_factor
            self.fl_y = self.meta["fl_y"] * self.scale_factor
        elif self.camera_angle_x is not None and self.image_width is not None:
            self.fl_x = self.image_width / (2.0 * np.tan(self.camera_angle_x / 2.0))
            self.fl_y = self.fl_x
        else:
            self.fl_x = None
            self.fl_y = None

        # Principal point
        if "cx" in self.meta and "cy" in self.meta:
            self.cx = self.meta["cx"] * self.scale_factor
            self.cy = self.meta["cy"] * self.scale_factor
        elif self.image_width is not None and self.image_height is not None:
            self.cx = self.image_width / 2.0
            self.cy = self.image_height / 2.0
        else:
            self.cx = None
            self.cy = None

    def get_intrinsics_matrix(self) -> np.ndarray:
        """Get camera intrinsics as 3x3 matrix"""
        if self.fl_x is None or self.fl_y is None:
            raise ValueError("Camera focal length not available")

        K = np.array([
            [self.fl_x, 0, self.cx],
            [0, self.fl_y, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample

        Returns:
            Dictionary containing:
                - image: RGB image tensor (H, W, 3) or None
                - transform_matrix: Camera pose matrix (4, 4)
                - intrinsics: Camera intrinsics matrix (3, 3)
                - file_path: Path to image file
        """
        frame = self.frames[idx]

        # Load or retrieve image
        if self.images is not None:
            image = self.images[idx]
        elif CV2_AVAILABLE and self.load_images:
            image = cv2.imread(str(frame["file_path"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.scale_factor != 1.0:
                new_size = (
                    int(image.shape[1] * self.scale_factor),
                    int(image.shape[0] * self.scale_factor)
                )
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            image = None

        # Convert to torch tensors
        sample = {
            "file_path": str(frame["file_path"]),
            "transform_matrix": torch.from_numpy(frame["transform_matrix"]),
            "intrinsics": torch.from_numpy(self.get_intrinsics_matrix()),
        }

        if image is not None:
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            sample["image"] = torch.from_numpy(image)

        return sample


def create_nerfstudio_dataloader(
    data_dir: Union[str, Path],
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    scale_factor: float = 1.0,
    load_images: bool = True
) -> DataLoader:
    """
    Create a DataLoader for NeRFStudio format data

    Args:
        data_dir: Path to dataset directory
        split: Dataset split
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        scale_factor: Image scale factor
        load_images: Whether to load images

    Returns:
        PyTorch DataLoader
    """
    dataset = NerfstudioDataset(
        data_dir=data_dir,
        split=split,
        scale_factor=scale_factor,
        load_images=load_images
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


def load_nerfstudio_metadata(data_dir: Union[str, Path]) -> Dict:
    """
    Load metadata from NeRFStudio transforms.json

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary containing metadata
    """
    transforms_file = Path(data_dir) / "transforms.json"

    if not transforms_file.exists():
        raise FileNotFoundError(f"transforms.json not found in {data_dir}")

    with open(transforms_file, "r") as f:
        meta = json.load(f)

    return meta


def convert_colmap_to_nerfstudio(
    colmap_dir: Union[str, Path],
    output_dir: Union[str, Path],
    images_dir: Optional[Union[str, Path]] = None
):
    """
    Convert COLMAP output to NeRFStudio format

    Args:
        colmap_dir: Path to COLMAP sparse reconstruction (text format)
        output_dir: Output directory for NeRFStudio format
        images_dir: Path to images directory (if different from colmap_dir/images)
    """
    # This is a placeholder - full implementation requires parsing COLMAP format
    # You can use nerfstudio's built-in converter: ns-process-data
    raise NotImplementedError(
        "Use NeRFStudio's built-in converter:\n"
        "ns-process-data images --data <path> --output-dir <output>\n"
        "Or for COLMAP data:\n"
        "ns-process-data colmap --data <path> --output-dir <output>"
    )
