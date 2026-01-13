"""
Temporal face tracking for reducing jitter and false detections.

Tracks faces across frames to ensure stability before displaying them.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackedFace:
    """
    A face being tracked across frames.

    Attributes:
        bbox: Current bounding box (x1, y1, x2, y2)
        identity: Recognized identity or None
        similarity: Recognition similarity score
        consecutive_frames: Number of consecutive frames detected
        frames_missing: Number of consecutive frames missing
        last_result: Last pipeline result for this face
    """
    bbox: Tuple[int, int, int, int]
    identity: Optional[str]
    similarity: float
    consecutive_frames: int = 0
    frames_missing: int = 0
    last_result: object = None  # PipelineResult

    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        """Calculate area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class FaceTracker:
    """
    Temporal face tracker for reducing jitter and false detections.

    Implements temporal filtering by:
    1. Requiring faces to be detected N consecutive frames before showing
    2. Keeping faces "alive" for M frames if temporarily lost
    3. Matching faces across frames using IoU (Intersection over Union)
    """

    def __init__(
        self,
        min_frames_to_show: int = 3,
        max_frames_missing: int = 5,
        iou_threshold: float = 0.3
    ):
        """
        Initialize face tracker.

        Args:
            min_frames_to_show: Minimum consecutive frames before showing face
            max_frames_missing: Maximum frames to keep face alive when missing
            iou_threshold: IoU threshold for matching faces across frames
        """
        self.min_frames_to_show = min_frames_to_show
        self.max_frames_missing = max_frames_missing
        self.iou_threshold = iou_threshold

        # Active tracked faces
        self.tracked_faces: List[TrackedFace] = []

        logger.info(
            f"FaceTracker initialized: "
            f"min_frames={min_frames_to_show}, "
            f"max_missing={max_frames_missing}, "
            f"iou_threshold={iou_threshold}"
        )

    def update(self, current_results: List) -> List:
        """
        Update tracker with current frame detections.

        Args:
            current_results: List of PipelineResult from current frame

        Returns:
            List of confirmed PipelineResult (only stable faces)
        """
        # Match current detections to tracked faces
        matched_indices = set()
        unmatched_results = []

        # Try to match each current detection with existing tracked faces
        for result in current_results:
            bbox = result.detection.bbox
            best_match_idx = None
            best_iou = 0.0

            # Find best matching tracked face
            for idx, tracked in enumerate(self.tracked_faces):
                if idx in matched_indices:
                    continue  # Already matched

                iou = self._calculate_iou(bbox, tracked.bbox)

                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            if best_match_idx is not None:
                # Update existing tracked face
                tracked = self.tracked_faces[best_match_idx]
                tracked.bbox = bbox
                tracked.identity = result.identity
                tracked.similarity = result.similarity
                tracked.consecutive_frames += 1
                tracked.frames_missing = 0
                tracked.last_result = result
                matched_indices.add(best_match_idx)
            else:
                # New unmatched detection
                unmatched_results.append(result)

        # Add new tracked faces for unmatched detections
        for result in unmatched_results:
            tracked = TrackedFace(
                bbox=result.detection.bbox,
                identity=result.identity,
                similarity=result.similarity,
                consecutive_frames=1,
                frames_missing=0,
                last_result=result
            )
            self.tracked_faces.append(tracked)

        # Update missing counts for unmatched tracked faces
        unmatched_tracked_indices = set(range(len(self.tracked_faces))) - matched_indices
        for idx in unmatched_tracked_indices:
            self.tracked_faces[idx].frames_missing += 1

        # Remove tracked faces that have been missing too long
        self.tracked_faces = [
            tracked for tracked in self.tracked_faces
            if tracked.frames_missing <= self.max_frames_missing
        ]

        # Return only confirmed faces (met min_frames_to_show threshold)
        confirmed_results = []
        for tracked in self.tracked_faces:
            if tracked.consecutive_frames >= self.min_frames_to_show:
                # Only show faces that have been consistently detected
                if tracked.last_result is not None:
                    confirmed_results.append(tracked.last_result)

        logger.debug(
            f"Tracker: {len(current_results)} detections → "
            f"{len(self.tracked_faces)} tracked → "
            f"{len(confirmed_results)} confirmed"
        )

        return confirmed_results

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            IoU score [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            # No intersection
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    def reset(self):
        """Reset tracker, clearing all tracked faces."""
        self.tracked_faces.clear()
        logger.info("FaceTracker reset")

    def get_stats(self) -> Dict:
        """
        Get tracker statistics.

        Returns:
            Dictionary with tracker stats
        """
        confirmed_count = sum(
            1 for t in self.tracked_faces
            if t.consecutive_frames >= self.min_frames_to_show
        )

        return {
            'total_tracked': len(self.tracked_faces),
            'confirmed': confirmed_count,
            'pending': len(self.tracked_faces) - confirmed_count,
            'min_frames_to_show': self.min_frames_to_show,
            'max_frames_missing': self.max_frames_missing
        }
