"""
Face authentication engine for high-stakes decisions.

Uses multi-frame confidence aggregation and stability checks.
"""

from collections import deque
from dataclasses import dataclass
from loguru import logger


@dataclass
class AuthenticationResult:
    """Result of authentication check."""
    authenticated: bool
    reason: str
    confidence: float


class FaceAuthenticator:
    """
    Authentication engine using weighted confidence history.

    Uses hybrid approach:
    - Weighted average (recent frames matter more)
    - Minimum confidence threshold
    - Bbox stability check
    """

    def __init__(
        self,
        authorized_user: str,
        min_frames_required: int = 5,
        weighted_threshold: float = 0.75,
        min_single_frame: float = 0.60,
        max_bbox_movement: float = 100.0,
        history_size: int = 10
    ):
        """
        Initialize authenticator.

        Args:
            authorized_user: Username to authenticate
            min_frames_required: Frames needed before decision
            weighted_threshold: Weighted average threshold
            min_single_frame: Minimum confidence for any frame
            max_bbox_movement: Max pixels movement for stability
            history_size: Size of history buffer
        """
        self.authorized_user = authorized_user
        self.min_frames_required = min_frames_required
        self.weighted_threshold = weighted_threshold
        self.min_single_frame = min_single_frame
        self.max_bbox_movement = max_bbox_movement

        self.confidence_history = deque(maxlen=history_size)
        self.identity_history = deque(maxlen=history_size)
        self.bbox_history = deque(maxlen=5)

        logger.info(
            f"FaceAuthenticator initialized: user={authorized_user}, "
            f"threshold={weighted_threshold}, min_frames={min_frames_required}"
        )

    def update(self, result) -> AuthenticationResult:
        """
        Update with new frame result and check authentication.

        Args:
            result: PipelineResult from current frame

        Returns:
            AuthenticationResult with decision
        """
        # No detection or wrong user
        if not result or result.identity != self.authorized_user:
            self.reset()
            return AuthenticationResult(
                authenticated=False,
                reason="Not authorized user" if result else "No face detected",
                confidence=0.0
            )

        # Update history
        self.confidence_history.append(result.similarity)
        self.identity_history.append(result.identity)
        self.bbox_history.append(result.detection.bbox)

        # Need enough data
        if len(self.confidence_history) < self.min_frames_required:
            return AuthenticationResult(
                authenticated=False,
                reason=f"Gathering data ({len(self.confidence_history)}/{self.min_frames_required})",
                confidence=result.similarity
            )

        # Weighted average (recent frames matter more)
        recent_scores = list(self.confidence_history)[-self.min_frames_required:]
        weights = self._get_weights(len(recent_scores))
        weighted_avg = sum(s * w for s, w in zip(recent_scores, weights))

        # Check minimum confidence
        min_confidence = min(recent_scores)
        if min_confidence < self.min_single_frame:
            return AuthenticationResult(
                authenticated=False,
                reason=f"Low frame detected ({min_confidence:.2f})",
                confidence=weighted_avg
            )

        # Check bbox stability
        if len(self.bbox_history) >= 3:
            movement = self._calculate_bbox_movement()
            if movement > self.max_bbox_movement:
                return AuthenticationResult(
                    authenticated=False,
                    reason=f"Unstable position ({movement:.0f}px)",
                    confidence=weighted_avg
                )

        # Final check
        if weighted_avg >= self.weighted_threshold:
            logger.success(f"Authenticated: {self.authorized_user} (confidence: {weighted_avg:.3f})")
            return AuthenticationResult(
                authenticated=True,
                reason=f"Authenticated",
                confidence=weighted_avg
            )

        return AuthenticationResult(
            authenticated=False,
            reason=f"Confidence too low ({weighted_avg:.2f})",
            confidence=weighted_avg
        )

    def _get_weights(self, n: int) -> list:
        """Generate weights that sum to 1, favoring recent frames."""
        weights = [(i + 1) for i in range(n)]
        total = sum(weights)
        return [w / total for w in weights]

    def _calculate_bbox_movement(self) -> float:
        """Calculate average movement between consecutive bboxes."""
        if len(self.bbox_history) < 2:
            return 0.0

        movements = []
        for i in range(len(self.bbox_history) - 1):
            bbox1 = self.bbox_history[i]
            bbox2 = self.bbox_history[i + 1]

            # Calculate center points
            c1_x = (bbox1[0] + bbox1[2]) / 2
            c1_y = (bbox1[1] + bbox1[3]) / 2
            c2_x = (bbox2[0] + bbox2[2]) / 2
            c2_y = (bbox2[1] + bbox2[3]) / 2

            # Euclidean distance
            distance = ((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2) ** 0.5
            movements.append(distance)

        return sum(movements) / len(movements)

    def reset(self):
        """Reset authentication state."""
        self.confidence_history.clear()
        self.identity_history.clear()
        self.bbox_history.clear()

    def get_stats(self) -> dict:
        """Get current authentication statistics."""
        if len(self.confidence_history) == 0:
            return {
                'frames_collected': 0,
                'avg_confidence': 0.0,
                'ready': False
            }

        return {
            'frames_collected': len(self.confidence_history),
            'avg_confidence': sum(self.confidence_history) / len(self.confidence_history),
            'min_confidence': min(self.confidence_history),
            'max_confidence': max(self.confidence_history),
            'ready': len(self.confidence_history) >= self.min_frames_required
        }
