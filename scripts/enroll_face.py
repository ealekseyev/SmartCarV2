#!/usr/bin/env python3
"""
Face enrollment utility script.

Easy-to-use tool for enrolling faces into the recognition database.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SmartCarNode
from loguru import logger


def enroll_face(name: str, config_path: str = "config/camera_config.yaml"):
    """
    Enroll a face interactively.

    Args:
        name: Name of person to enroll
        config_path: Path to config file
    """
    logger.info(f"Starting enrollment for: {name}")

    # Initialize system
    node = SmartCarNode(config_path=config_path)

    if not node.initialize():
        logger.error("Failed to initialize system")
        return False

    logger.info("=" * 60)
    logger.info("System initialized. Position your face in camera view.")
    logger.info("CONTROLS:")
    logger.info("  - Press 'c' to CAPTURE and enroll your face")
    logger.info("  - Press 'q' to QUIT without enrolling")
    logger.info("=" * 60)

    try:
        while True:
            # Read frame
            ret, frame = node.camera.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame")
                continue

            # Detect faces
            detections = node.detector.detect(frame)

            # Show preview
            from src.pipeline.face_pipeline import PipelineResult
            preview_results = []
            for det in detections:
                face_crop = node.detector.crop_face(frame, det)
                preview_results.append(
                    PipelineResult(
                        detection=det,
                        identity=f"[{name}]",
                        similarity=0.0,
                        face_crop=face_crop
                    )
                )

            node.viewer.show(
                frame,
                preview_results,
                0.0,
                {"Mode": "Enrollment", "Name": name, "Faces": len(detections)}
            )

            # Wait for key
            key = node.viewer.wait_key(1)

            if key == ord('c'):
                # Enroll
                if len(detections) == 0:
                    logger.warning("No faces detected! Try again.")
                    continue

                logger.info("Capturing and enrolling face...")
                success = node.pipeline.enroll_from_frame(frame, name)

                if success:
                    logger.success(f"Successfully enrolled {name}!")

                    # Save to individual user file
                    faces_dir = "data/faces"
                    node.recognizer.save_user_embedding(name, node.recognizer.face_database[name], faces_dir)
                    logger.info("Enrollment complete. Exiting...")

                    # Brief pause to show message
                    import time
                    time.sleep(1)

                    return True
                else:
                    logger.error("Enrollment failed. Try again (press 'c' to retry, 'q' to quit).")
                    continue

            elif key == ord('q') or key == 27:
                logger.info("Enrollment cancelled")
                return False

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return False
    finally:
        node.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Enroll a face into the database")
    parser.add_argument('name', type=str, help='Name of person to enroll')
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    success = enroll_face(args.name, args.config)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
