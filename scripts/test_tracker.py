#!/usr/bin/env python3
"""
Test script for face tracker temporal filtering.

This script simulates face tracking scenarios to verify the tracker works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.face_tracker import FaceTracker, TrackedFace
from src.pipeline.face_detector import FaceDetection
from src.pipeline.face_pipeline import PipelineResult
import numpy as np


def create_mock_result(bbox, identity="test_person", similarity=0.9):
    """Create a mock PipelineResult for testing."""
    detection = FaceDetection(
        bbox=bbox,
        confidence=0.95,
        landmarks=None
    )

    face_crop = np.zeros((112, 112, 3), dtype=np.uint8)

    return PipelineResult(
        detection=detection,
        identity=identity,
        similarity=similarity,
        face_crop=face_crop
    )


def test_consecutive_frames():
    """Test that faces require N consecutive frames before showing."""
    print("=" * 60)
    print("TEST 1: Consecutive Frames Requirement")
    print("=" * 60)

    tracker = FaceTracker(min_frames_to_show=3, max_frames_missing=5)

    # Frame 1: New face appears
    result1 = create_mock_result(bbox=(100, 100, 200, 200))
    confirmed = tracker.update([result1])
    print(f"Frame 1: {len(confirmed)} confirmed (expected 0)")
    assert len(confirmed) == 0, "Face should not show on first frame"

    # Frame 2: Same face appears again
    result2 = create_mock_result(bbox=(102, 102, 202, 202))  # Slightly moved
    confirmed = tracker.update([result2])
    print(f"Frame 2: {len(confirmed)} confirmed (expected 0)")
    assert len(confirmed) == 0, "Face should not show on second frame"

    # Frame 3: Same face appears third time
    result3 = create_mock_result(bbox=(104, 104, 204, 204))  # Slightly moved
    confirmed = tracker.update([result3])
    print(f"Frame 3: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Face should show on third consecutive frame"

    print("✓ Test passed: Consecutive frames requirement works!\n")


def test_persistence_when_missing():
    """Test that faces persist for N frames when temporarily lost."""
    print("=" * 60)
    print("TEST 2: Persistence When Missing")
    print("=" * 60)

    tracker = FaceTracker(min_frames_to_show=2, max_frames_missing=3)

    # Build up to showing face
    result1 = create_mock_result(bbox=(100, 100, 200, 200))
    tracker.update([result1])
    result2 = create_mock_result(bbox=(102, 102, 202, 202))
    confirmed = tracker.update([result2])
    assert len(confirmed) == 1, "Face should be showing"
    print(f"Initial: Face is showing")

    # Face disappears for 1 frame
    confirmed = tracker.update([])  # Empty frame
    print(f"Missing frame 1: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Face should persist for 1 missing frame"

    # Face disappears for 2nd frame
    confirmed = tracker.update([])
    print(f"Missing frame 2: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Face should persist for 2 missing frames"

    # Face disappears for 3rd frame
    confirmed = tracker.update([])
    print(f"Missing frame 3: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Face should persist for 3 missing frames"

    # Face disappears for 4th frame (exceeds max_frames_missing)
    confirmed = tracker.update([])
    print(f"Missing frame 4: {len(confirmed)} confirmed (expected 0)")
    assert len(confirmed) == 0, "Face should be removed after exceeding max_frames_missing"

    print("✓ Test passed: Persistence when missing works!\n")


def test_false_detection_filtering():
    """Test that brief false detections are filtered out."""
    print("=" * 60)
    print("TEST 3: False Detection Filtering")
    print("=" * 60)

    tracker = FaceTracker(min_frames_to_show=3, max_frames_missing=2)

    # Real face appears consistently
    real_face = create_mock_result(bbox=(100, 100, 200, 200), identity="real_person")
    tracker.update([real_face])
    tracker.update([real_face])
    confirmed = tracker.update([real_face])
    assert len(confirmed) == 1, "Real face should show after 3 frames"
    print(f"Real face established: {len(confirmed)} faces showing")

    # False detection appears for just 1 frame
    false_face = create_mock_result(bbox=(300, 300, 400, 400), identity="false_detection")
    confirmed = tracker.update([real_face, false_face])
    print(f"False detection frame 1: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Only real face should show, not false detection"

    # False detection gone, real face continues
    confirmed = tracker.update([real_face])
    print(f"After false detection gone: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Only real face should persist"

    print("✓ Test passed: False detections are filtered out!\n")


def test_multiple_faces():
    """Test tracking multiple faces simultaneously."""
    print("=" * 60)
    print("TEST 4: Multiple Face Tracking")
    print("=" * 60)

    tracker = FaceTracker(min_frames_to_show=2, max_frames_missing=2)

    # Two faces appear
    face1 = create_mock_result(bbox=(100, 100, 200, 200), identity="person1")
    face2 = create_mock_result(bbox=(300, 300, 400, 400), identity="person2")

    # Frame 1
    tracker.update([face1, face2])
    # Frame 2
    confirmed = tracker.update([face1, face2])
    print(f"Frame 2: {len(confirmed)} confirmed (expected 2)")
    assert len(confirmed) == 2, "Both faces should show after 2 consecutive frames"

    # One face disappears
    confirmed = tracker.update([face1])
    print(f"Face 2 missing: {len(confirmed)} confirmed (expected 2)")
    assert len(confirmed) == 2, "Both faces should persist (1 present, 1 in grace period)"

    # Face 2 stays missing too long
    tracker.update([face1])
    tracker.update([face1])
    confirmed = tracker.update([face1])
    print(f"Face 2 gone: {len(confirmed)} confirmed (expected 1)")
    assert len(confirmed) == 1, "Only face 1 should remain"

    print("✓ Test passed: Multiple face tracking works!\n")


def test_iou_matching():
    """Test IoU-based face matching across frames."""
    print("=" * 60)
    print("TEST 5: IoU Face Matching")
    print("=" * 60)

    tracker = FaceTracker(min_frames_to_show=1, max_frames_missing=5, iou_threshold=0.3)

    # Face in frame 1
    face1 = create_mock_result(bbox=(100, 100, 200, 200))
    confirmed = tracker.update([face1])
    stats = tracker.get_stats()
    print(f"Frame 1: {stats['total_tracked']} tracked")

    # Face moves slightly in frame 2 (should match)
    face2 = create_mock_result(bbox=(110, 110, 210, 210))
    confirmed = tracker.update([face2])
    stats = tracker.get_stats()
    print(f"Frame 2 (slight move): {stats['total_tracked']} tracked (expected 1)")
    assert stats['total_tracked'] == 1, "Should match as same face (good IoU)"

    # Face moves far in frame 3 (should create new track)
    face3 = create_mock_result(bbox=(400, 400, 500, 500))
    confirmed = tracker.update([face3])
    stats = tracker.get_stats()
    print(f"Frame 3 (large move): {stats['total_tracked']} tracked (expected 2)")
    assert stats['total_tracked'] == 2, "Should create new track (poor IoU)"

    print("✓ Test passed: IoU matching works correctly!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FACE TRACKER UNIT TESTS")
    print("=" * 60 + "\n")

    try:
        test_consecutive_frames()
        test_persistence_when_missing()
        test_false_detection_filtering()
        test_multiple_faces()
        test_iou_matching()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
