#!/usr/bin/env python3
"""
Test face recognition script.

Runs continuous face recognition and shows probability/confidence for each detected face.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SmartCarNode
from loguru import logger
import argparse


def test_recognition():
    """
    Run continuous face recognition test.

    Uses authorized users from recognition_config.yaml
    """
    logger.info("=" * 60)
    logger.info("Face Recognition Test")
    logger.info("=" * 60)

    # Initialize system (will auto-load authorized users)
    node = SmartCarNode()

    if not node.initialize():
        logger.error("Failed to initialize system")
        return False

    enrolled_faces = list(node.recognizer.face_database.keys())

    if len(enrolled_faces) == 0:
        logger.error("No authorized users loaded!")
        logger.error("Please:")
        logger.error("  1. Enroll a face: python scripts/enroll_face_advanced.py 'username'")
        logger.error("  2. Add username to authorized_users in config/recognition_config.yaml")
        return False

    logger.success(f"Loaded {len(enrolled_faces)} authorized user(s): {enrolled_faces}")

    logger.info("=" * 60)
    logger.info("Starting recognition test...")
    logger.info("The viewer will show:")
    logger.info("  - GREEN box = Recognized face (match)")
    logger.info("  - RED box = Unknown face (no match)")
    logger.info("  - Confidence score = Similarity (0.0 to 1.0)")
    logger.info("")
    logger.info("CONTROLS:")
    logger.info("  - Press 'q' to QUIT")
    logger.info("=" * 60)

    try:
        # Run the main loop
        node.run()
        return True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return True
    finally:
        node.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Test face recognition")

    args = parser.parse_args()

    success = test_recognition()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
