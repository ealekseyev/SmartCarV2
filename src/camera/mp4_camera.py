"""
MP4 video file camera simulator.

Plays back pre-recorded video as if it were a live camera feed.
Useful for testing without physical camera hardware.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from .camera_base import CameraBase, CameraSpecs


class MP4Camera(CameraBase):
    """
    Camera simulator that plays back MP4/video files.

    Acts as a drop-in replacement for USB camera.
    """

    def __init__(
        self,
        specs: CameraSpecs,
        video_path: str = "data/test_video.mp4",
        loop: bool = True
    ):
        """
        Initialize MP4 camera.

        Args:
            specs: Camera specifications (resolution, fps, format)
            video_path: Path to MP4/video file
            loop: Loop video when it reaches the end
        """
        super().__init__(specs)
        self.video_path = video_path
        self.loop = loop
        self.cap = None

    def open(self) -> bool:
        """Open video file."""
        try:
            logger.info(f"Opening video file: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.video_path}")
                return False

            # Get video properties
            vid_fps = self.cap.get(cv2.CAP_PROP_FPS)
            vid_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video: {vid_width}x{vid_height} @ {vid_fps:.1f}fps, {frame_count} frames")
            logger.info(f"Output: {self.specs.width}x{self.specs.height} @ {self.specs.fps}fps")

            self._is_opened = True
            return True

        except Exception as e:
            logger.error(f"Error opening MP4 camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from video file."""
        if not self._is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        # Loop video if enabled
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret:
            return False, None

        # Resize to match specs
        if frame.shape[1] != self.specs.width or frame.shape[0] != self.specs.height:
            frame = cv2.resize(frame, (self.specs.width, self.specs.height))

        # Convert to RGB if needed (OpenCV reads as BGR)
        if self.specs.format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return True, frame

    def release(self) -> None:
        """Release video file resources."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            logger.info("MP4 camera released")

    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self._is_opened

    def get_actual_fps(self) -> float:
        """Get actual FPS from video file."""
        if self.cap is not None and self._is_opened:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
