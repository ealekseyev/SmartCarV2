#!/usr/bin/env python3
"""
Quick camera test script.

Tests if camera subsystem is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera import USBCamera
from src.camera.camera_base import CameraSpecs
import cv2
from loguru import logger


def test_camera(device_id: int = 0):
    """
    Test camera functionality.

    Args:
        device_id: Camera device ID to test
    """
    logger.info(f"Testing camera {device_id}")

    # Create camera specs
    specs = CameraSpecs(
        width=1280,
        height=720,
        fps=30,
        format="RGB"
    )

    # Create camera
    camera = USBCamera(specs=specs, device_id=device_id)

    try:
        # Open camera
        if not camera.open():
            logger.error("Failed to open camera")
            return False

        logger.success("Camera opened successfully")
        logger.info(f"Camera properties: {camera.get_camera_properties()}")

        # Test frame capture
        logger.info("Capturing frames... Press 'q' to quit")

        frame_count = 0
        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                continue

            frame_count += 1

            # Convert RGB to BGR for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add info text
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | FPS: {camera.get_actual_fps():.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display
            cv2.imshow("Camera Test", display_frame)

            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break

        logger.success(f"Captured {frame_count} frames successfully")
        return True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return True

    except Exception as e:
        logger.error(f"Error during camera test: {e}")
        return False

    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera test complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test camera functionality")
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )

    args = parser.parse_args()

    success = test_camera(args.device)
    exit(0 if success else 1)
