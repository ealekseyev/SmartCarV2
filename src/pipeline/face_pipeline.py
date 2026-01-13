"""
Face recognition pipeline orchestrating detection and recognition.

Connects face detection → cropping → face recognition in a streamlined pipeline.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .face_detector import BlazeFaceDetector, FaceDetection
from .face_recognizer import MobileFaceNetRecognizer
from .face_tracker import FaceTracker


@dataclass
class PipelineResult:
    """Face recognition pipeline result."""
    detection: FaceDetection
    identity: Optional[str]  # None if not recognized
    similarity: float
    face_crop: np.ndarray  # Cropped face image


class FacePipeline:
    """
    End-to-end face recognition pipeline.

    Pipeline flow:
    1. Detect faces in frame (BlazeFace)
    2. Crop detected faces
    3. Generate embeddings and identify (MobileFaceNet)
    4. Return results with identity and confidence
    """

    def __init__(
        self,
        detector: BlazeFaceDetector,
        recognizer: MobileFaceNetRecognizer,
        min_face_size: int = 80,
        crop_padding: float = 0.2,
        enable_tracking: bool = True,
        min_frames_to_show: int = 3,
        max_frames_missing: int = 5,
        iou_threshold: float = 0.3
    ):
        """
        Initialize face pipeline.

        Args:
            detector: Face detector instance
            recognizer: Face recognizer instance
            min_face_size: Minimum face size to process (width or height in pixels)
            crop_padding: Padding around face crop (fraction of bbox)
            enable_tracking: Enable temporal tracking to reduce jitter
            min_frames_to_show: Consecutive frames required before showing face
            max_frames_missing: Frames to keep face alive when temporarily lost
            iou_threshold: IoU threshold for matching faces across frames
        """
        self.detector = detector
        self.recognizer = recognizer
        self.min_face_size = min_face_size
        self.crop_padding = crop_padding
        self.enable_tracking = enable_tracking

        # Initialize face tracker if enabled
        if enable_tracking:
            self.tracker = FaceTracker(
                min_frames_to_show=min_frames_to_show,
                max_frames_missing=max_frames_missing,
                iou_threshold=iou_threshold
            )
        else:
            self.tracker = None

        logger.info(
            f"Face pipeline initialized: "
            f"tracking={'ON' if enable_tracking else 'OFF'}, "
            f"min_size={min_face_size}px"
        )

    def process_frame(self, frame: np.ndarray) -> List[PipelineResult]:
        """
        Process a single frame through the full pipeline with temporal filtering.

        Args:
            frame: Input frame (RGB format)

        Returns:
            List of confirmed pipeline results (filtered by size and temporal stability)
        """
        results = []

        try:
            # Step 1: Detect faces
            detections = self.detector.detect(frame)

            if len(detections) == 0:
                logger.debug("No faces detected in frame")
                # Update tracker with empty detections
                if self.enable_tracking and self.tracker is not None:
                    return self.tracker.update([])
                return results

            logger.debug(f"Processing {len(detections)} detected faces")

            # Step 2 & 3: Process each detected face (with size filtering)
            for detection in detections:
                result = self._process_face(frame, detection)
                if result is not None:
                    results.append(result)

            # Step 4: Apply temporal tracking if enabled
            if self.enable_tracking and self.tracker is not None:
                results = self.tracker.update(results)
                logger.debug(
                    f"Temporal filtering: {len(detections)} detections → "
                    f"{len(results)} confirmed (after tracking)"
                )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

        return results

    def _process_face(
        self,
        frame: np.ndarray,
        detection: FaceDetection
    ) -> Optional[PipelineResult]:
        """
        Process a single detected face.

        Args:
            frame: Input frame
            detection: Face detection result

        Returns:
            PipelineResult or None if processing failed
        """
        try:
            # Filter small faces
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1

            if width < self.min_face_size or height < self.min_face_size:
                logger.debug(f"Face too small ({width}x{height}), skipping")
                return None

            # Crop face
            face_crop = self.detector.crop_face(frame, detection, padding=self.crop_padding)

            if face_crop.size == 0:
                logger.warning("Empty face crop, skipping")
                return None

            # Identify face
            identity, similarity = self.recognizer.identify(face_crop)

            # Create result
            result = PipelineResult(
                detection=detection,
                identity=identity,
                similarity=similarity,
                face_crop=face_crop
            )

            return result

        except Exception as e:
            logger.error(f"Error processing face: {e}")
            return None

    def enroll_from_frame(
        self,
        frame: np.ndarray,
        name: str,
        auto_select: bool = True
    ) -> bool:
        """
        Enroll a person from a frame.

        Args:
            frame: Input frame containing face to enroll
            name: Name to enroll
            auto_select: If True, automatically select largest face

        Returns:
            bool: True if enrollment successful
        """
        try:
            # Detect faces
            detections = self.detector.detect(frame)

            if len(detections) == 0:
                logger.error("No faces detected for enrollment")
                return False

            # Select face to enroll
            if auto_select:
                # Select largest face
                detection = max(
                    detections,
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                )
                logger.info("Auto-selected largest face for enrollment")
            else:
                if len(detections) > 1:
                    logger.warning(f"Multiple faces detected ({len(detections)}), using first")
                detection = detections[0]

            # Crop face
            face_crop = self.detector.crop_face(frame, detection, padding=self.crop_padding)

            # Enroll
            success = self.recognizer.enroll(name, face_crop)

            return success

        except Exception as e:
            logger.error(f"Error enrolling from frame: {e}")
            return False

    def get_statistics(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline stats
        """
        return {
            "detector_loaded": self.detector.is_loaded(),
            "recognizer_loaded": self.recognizer.is_loaded(),
            "enrolled_faces": self.recognizer.get_database_size(),
            "min_face_size": self.min_face_size,
            "crop_padding": self.crop_padding,
        }

    def is_ready(self) -> bool:
        """
        Check if pipeline is ready to process frames.

        Returns:
            bool: True if both detector and recognizer are loaded
        """
        detector_ready = self.detector.is_loaded()
        recognizer_ready = self.recognizer.is_loaded()

        if not detector_ready:
            logger.warning("Detector not loaded")
        if not recognizer_ready:
            logger.warning("Recognizer not loaded")

        return detector_ready and recognizer_ready
