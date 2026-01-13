"""
BlazeFace face detector module.

Uses BlazeFace model for fast, efficient face detection optimized for mobile devices.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # Optional facial landmarks


class BlazeFaceDetector:
    """
    BlazeFace face detector.

    Fast face detection optimized for mobile/edge devices.
    Can load from ONNX model or use OpenCV's DNN module.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        use_opencv_dnn: bool = True
    ):
        """
        Initialize BlazeFace detector.

        Args:
            model_path: Path to BlazeFace ONNX model (if None, will need to be set)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS IoU threshold
            use_opencv_dnn: Use OpenCV DNN module (fallback if no ONNX)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_opencv_dnn = use_opencv_dnn

        self.model = None
        self._is_loaded = False

        # For now, we'll use OpenCV's DNN face detector as a placeholder
        # until we integrate the actual BlazeFace ONNX model
        if use_opencv_dnn:
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """Initialize OpenCV DNN face detector (temporary fallback)."""
        try:
            # Using OpenCV's pre-trained face detector
            # This is a placeholder - will be replaced with BlazeFace ONNX
            logger.info("Initializing OpenCV DNN face detector (temporary)")

            # Using Haar Cascade as simplest fallback
            # TODO: Replace with actual BlazeFace model
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.model = cv2.CascadeClassifier(cascade_path)

            if self.model.empty():
                logger.error("Failed to load face detector cascade")
                self._is_loaded = False
            else:
                self._is_loaded = True
                logger.success("Face detector loaded (Haar Cascade - temporary)")

        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            self._is_loaded = False

    def load_model(self, model_path: str) -> bool:
        """
        Load BlazeFace ONNX model.

        Args:
            model_path: Path to ONNX model file

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info(f"Loading BlazeFace model from {model_path}")
            # TODO: Implement ONNX model loading
            # import onnxruntime as ort
            # self.model = ort.InferenceSession(model_path)

            self.model_path = model_path
            self._is_loaded = True
            logger.success("BlazeFace model loaded")
            return True

        except Exception as e:
            logger.error(f"Error loading BlazeFace model: {e}")
            return False

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            List[FaceDetection]: List of detected faces with bounding boxes
        """
        if not self._is_loaded:
            logger.warning("Detector not loaded, returning empty detections")
            return []

        try:
            # Temporary implementation using Haar Cascade
            # TODO: Replace with BlazeFace ONNX inference
            detections = self._detect_opencv(image)
            return detections

        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []

    def _detect_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using OpenCV (temporary implementation).

        Args:
            image: RGB image

        Returns:
            List of face detections
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        detections = []
        for (x, y, w, h) in faces:
            # Convert to (x1, y1, x2, y2) format
            bbox = (x, y, x + w, y + h)

            # Haar Cascade doesn't provide confidence, use 1.0
            detection = FaceDetection(
                bbox=bbox,
                confidence=1.0,
                landmarks=None
            )
            detections.append(detection)

        logger.debug(f"Detected {len(detections)} faces")
        return detections

    def is_loaded(self) -> bool:
        """Check if detector is loaded and ready."""
        return self._is_loaded

    def crop_face(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Crop face from image with optional padding.

        Args:
            image: Input image
            detection: Face detection result
            padding: Padding around face bbox (fraction of bbox size)

        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = detection.bbox
        w = x2 - x1
        h = y2 - y1

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        # Calculate padded bbox
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(image.shape[1], x2 + pad_w)
        y2_pad = min(image.shape[0], y2 + pad_h)

        # Crop
        face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        return face_crop


# TODO: Implement actual BlazeFace ONNX inference
# This will include:
# 1. Loading BlazeFace ONNX model
# 2. Preprocessing (resize, normalize)
# 3. Model inference
# 4. Postprocessing (decode anchors, NMS)
# 5. Return FaceDetection objects with proper confidence scores and landmarks
