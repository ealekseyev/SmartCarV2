"""
USB Camera implementation using OpenCV VideoCapture.

This implementation uses V4L2 (Video4Linux2) on Linux systems through OpenCV.
Guarantees the specified resolution and FPS as defined by CameraSpecs.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional
from threading import Lock
from loguru import logger

from .camera_base import CameraBase, CameraSpecs


class USBCamera(CameraBase):
    """
    USB RGB Camera implementation.

    Provides thread-safe access to USB camera with guaranteed resolution/FPS.
    Automatically handles format conversion (BGR to RGB if needed).
    """

    def __init__(self, specs: CameraSpecs, device_id: int = 0):
        """
        Initialize USB camera.

        Args:
            specs: Camera specifications (resolution, FPS, format)
            device_id: Camera device ID (default: 0 for /dev/video0)
        """
        super().__init__(specs)
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = Lock()

        # FPS measurement
        self._fps_counter = 0
        self._fps_start_time = 0
        self._measured_fps = 0.0

    def open(self) -> bool:
        """
        Open USB camera and configure to meet specifications.

        Returns:
            bool: True if camera opened and configured successfully
        """
        try:
            logger.info(f"Opening USB camera {self.device_id}")

            # Open camera device
            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False

            # Configure camera to meet specs
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.specs.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.specs.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.specs.fps)

            # Verify settings were applied
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")

            if actual_width != self.specs.width or actual_height != self.specs.height:
                logger.warning(
                    f"Camera resolution mismatch! "
                    f"Requested: {self.specs.width}x{self.specs.height}, "
                    f"Got: {actual_width}x{actual_height}"
                )

            if actual_fps != self.specs.fps:
                logger.warning(
                    f"Camera FPS mismatch! "
                    f"Requested: {self.specs.fps}, Got: {actual_fps}"
                )

            self._is_opened = True
            self._fps_start_time = time.time()
            logger.success("USB camera opened successfully")
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from USB camera with guaranteed specifications.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - Success flag
                - Frame as (H, W, 3) RGB/BGR numpy array or None
        """
        if not self.is_opened():
            logger.warning("Attempted to read from unopened camera")
            return False, None

        with self._lock:
            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    return False, None

                # Verify frame dimensions match specs
                if frame.shape[:2] != (self.specs.height, self.specs.width):
                    # Resize if needed to guarantee resolution
                    frame = cv2.resize(
                        frame,
                        (self.specs.width, self.specs.height),
                        interpolation=cv2.INTER_LINEAR
                    )
                    logger.debug(f"Resized frame to {self.specs.width}x{self.specs.height}")

                # Convert BGR to RGB if needed (OpenCV uses BGR by default)
                if self.specs.format.upper() == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update FPS measurement
                self._update_fps()

                return True, frame

            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                return False, None

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            logger.info("Releasing USB camera")
            self.cap.release()
            self.cap = None
            self._is_opened = False
            logger.success("USB camera released")

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def get_actual_fps(self) -> float:
        """Get measured FPS."""
        return self._measured_fps

    def _update_fps(self) -> None:
        """Update FPS measurement."""
        self._fps_counter += 1

        # Calculate FPS every second
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._measured_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start_time = time.time()

    def get_camera_properties(self) -> dict:
        """
        Get current camera properties.

        Returns:
            dict: Camera properties including resolution, FPS, etc.
        """
        if not self.is_opened():
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "format": self.specs.format,
            "backend": self.cap.getBackendName(),
            "measured_fps": self._measured_fps,
        }
