#!/usr/bin/env python3
"""Test face authenticator logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.face_authenticator import FaceAuthenticator
from src.pipeline.face_detector import FaceDetection
from src.pipeline.face_pipeline import PipelineResult
import numpy as np


def create_result(identity="evan_nicholas", similarity=0.85, bbox=(100, 100, 200, 200)):
    """Create mock result."""
    detection = FaceDetection(bbox=bbox, confidence=0.95, landmarks=None)
    face_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    return PipelineResult(detection=detection, identity=identity, similarity=similarity, face_crop=face_crop)


def test_basic_authentication():
    """Test basic authentication flow."""
    print("TEST 1: Basic Authentication")
    auth = FaceAuthenticator("evan_nicholas", min_frames_required=5)

    # Feed 5 frames with high confidence
    for i in range(5):
        result = create_result(similarity=0.85)
        auth_result = auth.update(result)
        print(f"Frame {i+1}: {auth_result.reason} (conf={auth_result.confidence:.2f})")

    assert auth_result.authenticated, "Should authenticate after 5 frames"
    print("✓ Passed\n")


def test_wrong_user():
    """Test rejection of wrong user."""
    print("TEST 2: Wrong User Rejection")
    auth = FaceAuthenticator("evan_nicholas")

    result = create_result(identity="wrong_person", similarity=0.90)
    auth_result = auth.update(result)

    assert not auth_result.authenticated, "Should reject wrong user"
    print(f"Result: {auth_result.reason}")
    print("✓ Passed\n")


def test_low_confidence():
    """Test rejection of low confidence."""
    print("TEST 3: Low Confidence Rejection")
    auth = FaceAuthenticator("evan_nicholas", weighted_threshold=0.75)

    # Feed frames with low confidence
    for i in range(5):
        result = create_result(similarity=0.50)
        auth_result = auth.update(result)

    assert not auth_result.authenticated, "Should reject low confidence"
    print(f"Result: {auth_result.reason}")
    print("✓ Passed\n")


def test_unstable_bbox():
    """Test rejection of unstable bbox."""
    print("TEST 4: Unstable Bbox Rejection")
    auth = FaceAuthenticator("evan_nicholas", max_bbox_movement=50.0)

    # Feed frames with large movement
    for i in range(5):
        bbox = (100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50)
        result = create_result(similarity=0.85, bbox=bbox)
        auth_result = auth.update(result)

    assert not auth_result.authenticated, "Should reject unstable bbox"
    print(f"Result: {auth_result.reason}")
    print("✓ Passed\n")


def test_weighted_average():
    """Test weighted average calculation."""
    print("TEST 5: Weighted Average")
    auth = FaceAuthenticator("evan_nicholas", min_frames_required=5, weighted_threshold=0.75)

    # Feed increasing confidence (recent should matter more)
    confidences = [0.60, 0.65, 0.70, 0.80, 0.90]
    for conf in confidences:
        result = create_result(similarity=conf)
        auth_result = auth.update(result)

    print(f"Confidences: {confidences}")
    print(f"Weighted result: {auth_result.confidence:.3f}")
    print(f"Authenticated: {auth_result.authenticated}")
    assert auth_result.authenticated, "Should authenticate with high recent scores"
    print("✓ Passed\n")


def main():
    print("=" * 60)
    print("FACE AUTHENTICATOR TESTS")
    print("=" * 60 + "\n")

    try:
        test_basic_authentication()
        test_wrong_user()
        test_low_confidence()
        test_unstable_bbox()
        test_weighted_average()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1


if __name__ == "__main__":
    exit(main())
