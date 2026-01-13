#!/usr/bin/env python3
"""
Advanced face enrollment with multiple samples.

Captures multiple face samples from different angles and lighting
to create a robust averaged embedding for better recognition accuracy.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SmartCarNode
from loguru import logger


def enroll_face_advanced(
    name: str,
    num_samples: int = 10,
    config_path: str = "config/camera_config.yaml"
):
    """
    Advanced face enrollment with multiple samples.

    Args:
        name: Name of person to enroll
        num_samples: Number of face samples to capture
        config_path: Path to config file
    """
    logger.info(f"Starting ADVANCED enrollment for: {name}")
    logger.info(f"Will capture {num_samples} samples of your face")

    # Initialize system
    node = SmartCarNode(config_path=config_path)

    if not node.initialize():
        logger.error("Failed to initialize system")
        return False

    logger.info("=" * 60)
    logger.info("ADVANCED ENROLLMENT MODE")
    logger.info("=" * 60)
    logger.info("This will capture multiple samples of your face for better accuracy.")
    logger.info("")
    logger.info("Instructions:")
    logger.info("  1. Position your face in the camera view")
    logger.info("  2. Press 'SPACE' when ready to start capturing")
    logger.info("  3. Slowly move your head slightly (left, right, up, down)")
    logger.info("  4. Keep your face in frame during capture")
    logger.info("  5. System will auto-capture samples every 0.5 seconds")
    logger.info("")
    logger.info("Press 'q' to cancel")
    logger.info("=" * 60)

    embeddings = []
    captured_samples = 0
    capturing = False

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

            if len(detections) > 0:
                for det in detections:
                    face_crop = node.detector.crop_face(frame, det)

                    if capturing:
                        label = f"[Capturing {captured_samples}/{num_samples}]"
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

            # Display
            extra_info = {
                "Mode": "Advanced Enrollment",
                "Name": name,
                "Samples": f"{captured_samples}/{num_samples}",
                "Faces": len(detections)
            }

            node.viewer.show(frame, preview_results, 0.0, extra_info)

            # Handle keys
            key = node.viewer.wait_key(1)

            if key == ord(' ') and not capturing:
                # Start capturing
                if len(detections) == 0:
                    logger.warning("No face detected! Position your face in view.")
                    continue

                logger.info("Starting capture sequence...")
                logger.info("Move your head slowly (left, right, up, down)")
                capturing = True
                last_capture_time = 0

            elif key == ord('q') or key == 27:
                logger.info("Enrollment cancelled")
                return False

            # Auto-capture samples
            if capturing and len(detections) > 0:
                current_time = time.time()

                if captured_samples == 0 or (current_time - last_capture_time >= 0.5):
                    # Capture this sample
                    detection = detections[0]  # Use first (largest) face
                    face_crop = node.detector.crop_face(frame, detection)

                    # Generate embedding
                    embedding = node.recognizer.get_embedding(face_crop)

                    if embedding is not None:
                        embeddings.append(embedding)
                        captured_samples += 1
                        last_capture_time = current_time

                        logger.info(f"Captured sample {captured_samples}/{num_samples}")

                        if captured_samples >= num_samples:
                            # Done capturing
                            logger.success(f"Captured all {num_samples} samples!")
                            break
                    else:
                        logger.warning("Failed to generate embedding, retrying...")

        if captured_samples < num_samples:
            logger.error("Insufficient samples captured")
            return False

        # Calculate averaged embedding
        logger.info("Processing captured samples...")
        embeddings_array = np.array(embeddings)

        # Calculate mean embedding
        mean_embedding = np.mean(embeddings_array, axis=0)

        # Normalize
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

        # Calculate embedding quality (variance)
        embedding_variance = np.var(embeddings_array, axis=0).mean()
        logger.info(f"Embedding variance: {embedding_variance:.6f}")

        # Check consistency
        similarities = []
        for emb in embeddings:
            sim = np.dot(mean_embedding, emb)
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)

        logger.info(f"Average self-similarity: {avg_similarity:.3f}")
        logger.info(f"Minimum self-similarity: {min_similarity:.3f}")

        if min_similarity < 0.5:
            logger.warning("Low consistency detected! Some samples may be poor quality.")
            logger.warning(f"Minimum similarity: {min_similarity:.3f}")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Enrollment cancelled")
                return False

        # Save to individual user file
        node.recognizer.face_database[name] = mean_embedding
        logger.success(f"Enrolled {name} with averaged embedding from {num_samples} samples")

        # Save to individual file
        faces_dir = "data/faces"
        node.recognizer.save_user_embedding(name, mean_embedding, faces_dir)

        logger.info("=" * 60)
        logger.success("ENROLLMENT COMPLETE!")
        logger.info(f"Quality metrics:")
        logger.info(f"  - Samples captured: {num_samples}")
        logger.info(f"  - Avg self-similarity: {avg_similarity:.3f}")
        logger.info(f"  - Embedding variance: {embedding_variance:.6f}")
        logger.info("=" * 60)

        return True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return False
    finally:
        node.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced face enrollment with multiple samples")
    parser.add_argument('name', type=str, help='Name of person to enroll')
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of face samples to capture (default: 10)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    success = enroll_face_advanced(args.name, args.samples, args.config)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
