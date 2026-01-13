#!/usr/bin/env python3
"""
Add more face samples to existing enrollment.

Use this to add samples in different lighting conditions, angles, etc.
This APPENDS to your existing embeddings for better accuracy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enroll_face_advanced import enroll_face_advanced
from src.pipeline.face_recognizer import MobileFaceNetRecognizer
import numpy as np
from loguru import logger


def add_samples(name: str, num_samples: int = 10):
    """
    Add more samples to existing user enrollment.

    Args:
        name: Username (must already be enrolled)
        num_samples: Number of additional samples to capture
    """
    faces_dir = "data/faces"
    filepath = Path(faces_dir) / f"{name}.npz"

    # Check if user exists
    if not filepath.exists():
        logger.error(f"User {name} not enrolled yet!")
        logger.error(f"First enroll with: python3 scripts/enroll_face_advanced.py '{name}'")
        return False

    # Load existing
    data = np.load(filepath)
    if 'embeddings' in data:
        existing_count = len(data['embeddings'])
    elif 'embedding' in data:
        existing_count = 1
    else:
        logger.error("Invalid embedding file format")
        return False

    logger.info(f"User {name} currently has {existing_count} embedding(s)")
    logger.info(f"Adding {num_samples} more samples...")
    logger.info("")
    logger.info("TIP: Try different conditions:")
    logger.info("  - Different lighting (bright, dim, natural light)")
    logger.info("  - Different angles (slight left/right)")
    logger.info("  - With/without glasses")
    logger.info("")

    # Use the advanced enrollment but in APPEND mode
    from src.main import SmartCarNode
    import time

    node = SmartCarNode()
    if not node.initialize():
        logger.error("Failed to initialize")
        return False

    logger.info("=" * 60)
    logger.info("ADDING MORE SAMPLES")
    logger.info("=" * 60)
    logger.info(f"Position your face (maybe in different lighting/angle)")
    logger.info("Press SPACE to start capturing")
    logger.info("=" * 60)

    embeddings = []
    captured_samples = 0
    capturing = False

    try:
        while True:
            ret, frame = node.camera.read()
            if not ret or frame is None:
                continue

            detections = node.detector.detect(frame)

            from src.pipeline.face_pipeline import PipelineResult
            preview_results = []

            if len(detections) > 0:
                for det in detections:
                    face_crop = node.detector.crop_face(frame, det)

                    if capturing:
                        label = f"[Adding {captured_samples}/{num_samples}]"
                    else:
                        label = f"[{name}] - Press SPACE"

                    preview_results.append(
                        PipelineResult(
                            detection=det,
                            identity=label,
                            similarity=0.0,
                            face_crop=face_crop
                        )
                    )

            extra_info = {
                "Mode": "Add Samples",
                "Name": name,
                "New": f"{captured_samples}/{num_samples}",
                "Existing": existing_count
            }

            node.viewer.show(frame, preview_results, 0.0, extra_info)

            key = node.viewer.wait_key(1)

            if key == ord(' ') and not capturing:
                if len(detections) == 0:
                    logger.warning("No face detected!")
                    continue
                logger.info("Starting capture...")
                capturing = True
                last_capture_time = 0

            elif key == ord('q') or key == 27:
                logger.info("Cancelled")
                return False

            if capturing and len(detections) > 0:
                current_time = time.time()

                if captured_samples == 0 or (current_time - last_capture_time >= 0.5):
                    detection = detections[0]
                    face_crop = node.detector.crop_face(frame, detection)

                    embedding = node.recognizer.get_embedding(face_crop)

                    if embedding is not None:
                        embeddings.append(embedding)
                        captured_samples += 1
                        last_capture_time = current_time
                        logger.info(f"Captured sample {captured_samples}/{num_samples}")

                        if captured_samples >= num_samples:
                            logger.success(f"Captured all {num_samples} new samples!")
                            break

        if captured_samples < num_samples:
            logger.error("Insufficient samples")
            return False

        # Convert to array and save with APPEND mode
        new_embeddings = np.array(embeddings)

        # Save - this will append
        node.recognizer.save_user_embedding(name, new_embeddings, faces_dir, append=True)

        logger.success(f"Added {num_samples} new samples to {name}")
        logger.info(f"Total embeddings for {name}: {existing_count + num_samples}")

        return True

    except KeyboardInterrupt:
        logger.info("Interrupted")
        return False
    finally:
        node.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add more face samples to existing enrollment")
    parser.add_argument('name', type=str, help='Username (must already be enrolled)')
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of additional samples to capture (default: 10)'
    )

    args = parser.parse_args()

    success = add_samples(args.name, args.samples)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
