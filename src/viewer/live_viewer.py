"""
Live viewer for displaying camera feed with face detection/recognition overlays.

Displays bounding boxes, identity labels, confidence scores, and FPS metrics.
"""

import cv2
import numpy as np
from typing import List, Optional
from loguru import logger

from src.pipeline.face_pipeline import PipelineResult


class LiveViewer:
    """
    Live display viewer with face recognition overlays.

    Shows:
    - Camera feed
    - Face bounding boxes
    - Identity labels
    - Confidence scores
    - FPS counter
    - System status
    """

    def __init__(
        self,
        window_name: str = "SmartCar Face Recognition",
        show_fps: bool = True,
        show_confidence: bool = True,
        bbox_color_known: tuple = (0, 255, 0),    # Green for known faces
        bbox_color_unknown: tuple = (0, 0, 255),  # Red for unknown faces
        bbox_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ):
        """
        Initialize live viewer.

        Args:
            window_name: OpenCV window name
            show_fps: Display FPS counter
            show_confidence: Display recognition confidence
            bbox_color_known: BGR color for known faces
            bbox_color_unknown: BGR color for unknown faces
            bbox_thickness: Bounding box line thickness
            font_scale: Font scale for text
            font_thickness: Font thickness for text
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.bbox_color_known = bbox_color_known
        self.bbox_color_unknown = bbox_color_unknown
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

        self._window_created = False

    def create_window(self):
        """Create OpenCV window."""
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
            logger.info(f"Created viewer window: {self.window_name}")

    def show(
        self,
        frame: np.ndarray,
        results: List[PipelineResult],
        fps: float = 0.0,
        extra_info: Optional[dict] = None
    ):
        """
        Display frame with overlays.

        Args:
            frame: Input frame (RGB format)
            results: List of pipeline results to overlay
            fps: Current FPS to display
            extra_info: Additional info to display
        """
        # Create window if needed
        if not self._window_created:
            self.create_window()

        # Convert RGB to BGR for OpenCV display
        display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

        # Draw overlays
        display_frame = self._draw_results(display_frame, results)
        display_frame = self._draw_info(display_frame, fps, extra_info)

        # Show frame
        cv2.imshow(self.window_name, display_frame)

    def _draw_results(
        self,
        frame: np.ndarray,
        results: List[PipelineResult]
    ) -> np.ndarray:
        """
        Draw face detection and recognition results on frame.

        Args:
            frame: Frame to draw on (BGR)
            results: Pipeline results

        Returns:
            Frame with overlays
        """
        for result in results:
            # Get bbox coordinates
            x1, y1, x2, y2 = result.detection.bbox

            # Choose color based on recognition
            if result.identity is not None:
                color = self.bbox_color_known
                label = result.identity
            else:
                color = self.bbox_color_unknown
                label = "Unknown"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.bbox_thickness
            )

            # Prepare label text
            if self.show_confidence:
                label_text = f"{label} ({result.similarity:.2f})"
            else:
                label_text = label

            # Calculate label background size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1  # Filled
            )

            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )

        return frame

    def _draw_info(
        self,
        frame: np.ndarray,
        fps: float,
        extra_info: Optional[dict]
    ) -> np.ndarray:
        """
        Draw info overlay (FPS, status, etc.).

        Args:
            frame: Frame to draw on
            fps: FPS value
            extra_info: Additional information

        Returns:
            Frame with info overlay
        """
        info_y = 30
        line_height = 25

        # Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 255, 0),
                self.font_thickness
            )
            info_y += line_height

        # Draw extra info
        if extra_info:
            for key, value in extra_info.items():
                info_text = f"{key}: {value}"

                # Color code state: green for UNLOCKED, red for LOCKED
                if key == "State":
                    if "UNLOCKED" in str(value):
                        color = (0, 255, 0)  # Green
                    else:
                        color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 255)  # White for other info

                cv2.putText(
                    frame,
                    info_text,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale * 0.8,
                    color,
                    self.font_thickness - 1
                )
                info_y += line_height

        return frame

    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for key press.

        Args:
            delay: Delay in milliseconds

        Returns:
            Key code (or -1 if no key pressed)
        """
        return cv2.waitKey(delay)

    def destroy(self):
        """Destroy viewer window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
            logger.info(f"Destroyed viewer window: {self.window_name}")

    def is_window_open(self) -> bool:
        """Check if window is still open."""
        if not self._window_created:
            return False

        try:
            # Check if window exists
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        self.create_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy()
