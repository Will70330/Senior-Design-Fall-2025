"""
Video Frame Extraction Module
Extracts optimal frames from high frame rate video for Structure from Motion
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract optimal frames from video for SfM reconstruction"""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        similarity_threshold: float = 0.95,
        min_frame_interval: int = 5,
        max_frames: Optional[int] = None
    ):
        """
        Args:
            blur_threshold: Laplacian variance threshold for blur detection (higher = sharper)
            similarity_threshold: SSIM threshold for frame similarity (lower = more different)
            min_frame_interval: Minimum frames between selected frames
            max_frames: Maximum number of frames to extract (None = no limit)
        """
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval
        self.max_frames = max_frames

    def calculate_blur_score(self, frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using MSE"""
        # Resize for faster comparison
        size = (256, 256)
        f1 = cv2.resize(frame1, size)
        f2 = cv2.resize(frame2, size)

        # Convert to grayscale
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # Compute mean squared error
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)

        # Convert to similarity score (0 = identical, higher = more different)
        max_pixel_value = 255.0
        similarity = 1 - (mse / (max_pixel_value ** 2))

        return similarity

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_prefix: str = "frame"
    ) -> List[str]:
        """
        Extract optimal frames from video

        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            frame_prefix: Prefix for output frame filenames

        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video info: {total_frames} frames @ {fps:.2f} fps")

        selected_frames = []
        frame_paths = []
        last_selected_frame = None
        frame_count = 0
        frames_since_last = 0

        with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frames_since_last += 1
                pbar.update(1)

                # Check frame interval
                if frames_since_last < self.min_frame_interval:
                    continue

                # Check blur
                blur_score = self.calculate_blur_score(frame)
                if blur_score < self.blur_threshold:
                    continue

                # Check similarity with last selected frame
                if last_selected_frame is not None:
                    similarity = self.calculate_similarity(frame, last_selected_frame)
                    if similarity > self.similarity_threshold:
                        continue

                # Frame passed all checks - save it
                frame_num = len(selected_frames)
                output_path = output_dir / f"{frame_prefix}_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                selected_frames.append(frame.copy())
                frame_paths.append(str(output_path))
                last_selected_frame = frame.copy()
                frames_since_last = 0

                logger.debug(f"Selected frame {frame_num} (blur={blur_score:.2f})")

                # Check max frames limit
                if self.max_frames and len(selected_frames) >= self.max_frames:
                    logger.info(f"Reached max frames limit: {self.max_frames}")
                    break

        cap.release()

        logger.info(f"Extracted {len(selected_frames)} frames from {total_frames} total frames")
        logger.info(f"Frames saved to: {output_dir}")

        return frame_paths


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    blur_threshold: float = 100.0,
    similarity_threshold: float = 0.95,
    min_frame_interval: int = 5,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Convenience function to extract frames from video

    Args:
        video_path: Path to input video
        output_dir: Directory for output frames
        blur_threshold: Minimum blur score (higher = sharper required)
        similarity_threshold: Maximum similarity with previous frame (lower = more different required)
        min_frame_interval: Minimum frames between selections
        max_frames: Maximum frames to extract

    Returns:
        List of frame file paths
    """
    extractor = FrameExtractor(
        blur_threshold=blur_threshold,
        similarity_threshold=similarity_threshold,
        min_frame_interval=min_frame_interval,
        max_frames=max_frames
    )

    return extractor.extract_frames(video_path, output_dir)
