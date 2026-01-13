"""
Base camera interface defining the contract for all camera implementations.

This abstract class ensures that any camera implementation guarantees:
- Specific resolution output
- Minimum frame rate
- Consistent frame format (RGB/BGR)
- Thread-safe frame access
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraSpecs:
    """Camera specification requirements."""
    width: int
    height: int
    fps: int
    format: str = "RGB"  # RGB or BGR

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


class CameraBase(ABC):
    """
    Abstract base class for camera implementations.

    All camera classes must inherit from this and implement the required methods.
    This ensures modularity - cameras can be swapped without changing pipeline code.
    """

    def __init__(self, specs: CameraSpecs):
        """
        Initialize camera with specifications.

        Args:
            specs: CameraSpecs object defining required resolution, FPS, and format
        """
        self.specs = specs
        self._is_opened = False

    @abstractmethod
    def open(self) -> bool:
        """
        Open and initialize the camera hardware.

        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        The frame MUST meet the specifications defined in self.specs:
        - Resolution: (specs.width, specs.height)
        - Format: specs.format (RGB or BGR)
        - Type: np.ndarray with dtype uint8

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - Success flag (True if frame captured successfully)
                - Frame as numpy array (H, W, 3) or None if failed
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release camera resources and close connection.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if camera is currently opened and available.

        Returns:
            bool: True if camera is opened and ready
        """
        pass

    @abstractmethod
    def get_actual_fps(self) -> float:
        """
        Get the actual measured frame rate.

        Returns:
            float: Actual FPS being delivered by the camera
        """
        pass

    def get_specs(self) -> CameraSpecs:
        """
        Get camera specifications.

        Returns:
            CameraSpecs: The camera specification requirements
        """
        return self.specs

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
