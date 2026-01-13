#!/usr/bin/env python3
"""
Multi-pose face enrollment with automatic sample capture.

Captures 10 frontal samples and 3 samples per additional pose (left, right, up, down)
to create robust embeddings for better recognition accuracy across different angles.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SmartCarNode
from loguru import logger


# Pose configurations
POSE_CONFIG = {
    'frontal': {
        'samples': 10,
        'instruction': 'Look straight at the camera',
        'color': (0, 255, 0)  # Green
    },
    'left': {
        'samples': 3,
        'instruction': 'Turn your head LEFT (look left)',
        'color': (255, 165, 0)  # Orange
    },
    'right': {
        'samples': 3,
        'instruction': 'Turn your head RIGHT (look right)',
        'color': (255, 165, 0)  # Orange
    },
    'up': {
        'samples': 3,
        'instruction': 'Tilt your head UP (look up)',
        'color': (255, 165, 0)  # Orange
    },
    'down': {
        'samples': 3,
        'instruction': 'Tilt your head DOWN (look down)',
        'color': (255, 165, 0)  # Orange
    }
}


def capture_pose_samples(
    node: SmartCarNode,
    name: str,
    pose: str,
    num_samples: int,
    instruction: str
) -> list:
    """
    Capture samples for a specific pose.

    Args:
        node: SmartCarNode instance
        name: Person's name
        pose: Pose name (frontal, left, right, up, down)
        num_samples: Number of samples to capture
        instruction: Instruction to show user

    Returns:
        List of embeddings
    """
    logger.info(f"=" * 60)
    logger.info(f"POSE: {pose.upper()} ({num_samples} samples)")
    logger.info(f"=" * 60)
    logger.info(f"Instruction: {instruction}")
    logger.info("")
    logger.info("Controls:")
    logger.info("  - Press 'SPACE' to start capturing")
    logger.info("  - Press 'q' to cancel enrollment")
    logger.info("=" * 60)

    embeddings = []
    captured_samples = 0
    capturing = False
    last_capture_time = 0

    try:
        while captured_samples < num_samples:
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
                        label = f"[{pose.upper()} {captured_samples}/{num_samples}]"
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
                "Mode": "Multi-Pose Enrollment",
                "Name": name,
                "Pose": f"{pose.upper()} ({captured_samples}/{num_samples})",
                "Instruction": instruction,
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

                logger.info(f"Starting capture for {pose} pose...")
                logger.info(instruction)
                capturing = True
                last_capture_time = 0

            elif key == ord('q') or key == 27:
                logger.info("Enrollment cancelled by user")
                return None

            # Auto-capture samples
            if capturing and len(detections) > 0:
                current_time = time.time()

                if captured_samples == 0 or (current_time - last_capture_time >= 0.5):
                    # Capture this sample
                    # Select largest face
                    detection = max(
                        detections,
                        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                    )
                    face_crop = node.detector.crop_face(frame, detection)

                    # Generate embedding
                    embedding = node.recognizer.get_embedding(face_crop)

                    if embedding is not None:
                        embeddings.append(embedding)
                        captured_samples += 1
                        last_capture_time = current_time

                        logger.info(f"✓ Captured {pose} sample {captured_samples}/{num_samples}")

                        if captured_samples >= num_samples:
                            logger.success(f"Completed {pose} pose!")
                            break
                    else:
                        logger.warning("Failed to generate embedding, retrying...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return None

    return embeddings


def calculate_averaged_embedding(embeddings: list, pose: str) -> tuple:
    """
    Calculate averaged embedding and quality metrics.

    Args:
        embeddings: List of embeddings
        pose: Pose name

    Returns:
        Tuple of (averaged_embedding, quality_metrics)
    """
    embeddings_array = np.array(embeddings)

    # Calculate mean embedding
    mean_embedding = np.mean(embeddings_array, axis=0)

    # Normalize
    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

    # Calculate quality metrics
    embedding_variance = np.var(embeddings_array, axis=0).mean()

    # Check consistency
    similarities = []
    for emb in embeddings:
        sim = np.dot(mean_embedding, emb)
        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)

    quality_metrics = {
        'variance': embedding_variance,
        'avg_similarity': avg_similarity,
        'min_similarity': min_similarity,
        'num_samples': len(embeddings)
    }

    logger.info(f"{pose} pose quality:")
    logger.info(f"  - Samples: {len(embeddings)}")
    logger.info(f"  - Avg similarity: {avg_similarity:.3f}")
    logger.info(f"  - Min similarity: {min_similarity:.3f}")
    logger.info(f"  - Variance: {embedding_variance:.6f}")

    if min_similarity < 0.5:
        logger.warning(f"Low consistency for {pose} pose (min similarity: {min_similarity:.3f})")

    return mean_embedding, quality_metrics


def enroll_face_multipose(
    name: str,
    config_path: str = "config/camera_config.yaml"
):
    """
    Multi-pose face enrollment.

    Args:
        name: Name of person to enroll
        config_path: Path to config file

    Returns:
        bool: True if enrollment successful
    """
    logger.info(f"Starting MULTI-POSE enrollment for: {name}")
    logger.info("")

    # Initialize system
    node = SmartCarNode(config_path=config_path)

    if not node.initialize():
        logger.error("Failed to initialize system")
        return False

    logger.info("=" * 60)
    logger.info("MULTI-POSE ENROLLMENT")
    logger.info("=" * 60)
    logger.info("This will capture face samples from multiple angles:")
    logger.info("  1. FRONTAL: 10 samples (straight ahead)")
    logger.info("  2. LEFT: 3 samples (head turned left)")
    logger.info("  3. RIGHT: 3 samples (head turned right)")
    logger.info("  4. UP: 3 samples (head tilted up)")
    logger.info("  5. DOWN: 3 samples (head tilted down)")
    logger.info("")
    logger.info("Total: 22 samples across 5 poses")
    logger.info("=" * 60)
    logger.info("")

    all_embeddings = {}
    all_quality_metrics = {}

    try:
        # Capture each pose
        for pose, config in POSE_CONFIG.items():
            embeddings = capture_pose_samples(
                node=node,
                name=name,
                pose=pose,
                num_samples=config['samples'],
                instruction=config['instruction']
            )

            if embeddings is None:
                logger.error(f"Failed to capture {pose} pose")
                return False

            # Calculate averaged embedding
            avg_embedding, quality_metrics = calculate_averaged_embedding(embeddings, pose)

            all_embeddings[pose] = avg_embedding
            all_quality_metrics[pose] = quality_metrics

            logger.success(f"✓ {pose.upper()} pose complete")
            logger.info("")

        # Check overall quality
        logger.info("=" * 60)
        logger.info("ENROLLMENT SUMMARY")
        logger.info("=" * 60)

        all_good = True
        for pose, metrics in all_quality_metrics.items():
            logger.info(f"{pose.upper()}:")
            logger.info(f"  Samples: {metrics['num_samples']}")
            logger.info(f"  Avg similarity: {metrics['avg_similarity']:.3f}")
            logger.info(f"  Min similarity: {metrics['min_similarity']:.3f}")

            if metrics['min_similarity'] < 0.5:
                logger.warning(f"  ⚠ Low quality detected for {pose}")
                all_good = False

        logger.info("=" * 60)

        if not all_good:
            logger.warning("Some poses have low quality samples.")
            response = input("Continue with enrollment anyway? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Enrollment cancelled")
                return False

        # Save all embeddings
        faces_dir = "data/faces"
        logger.info(f"Saving embeddings to {faces_dir}/{name}/...")

        for pose, embedding in all_embeddings.items():
            success = node.recognizer.save_user_embedding(name, embedding, pose, faces_dir)
            if not success:
                logger.error(f"Failed to save {pose} embedding")
                return False

        logger.info("=" * 60)
        logger.success("ENROLLMENT COMPLETE!")
        logger.info(f"User: {name}")
        logger.info(f"Poses captured: {len(all_embeddings)}")
        logger.info(f"Total samples: {sum(m['num_samples'] for m in all_quality_metrics.values())}")
        logger.info(f"Location: {faces_dir}/{name}/")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Files created:")
        for pose in all_embeddings.keys():
            logger.info(f"  - {faces_dir}/{name}/{pose}.npz")
        logger.info("=" * 60)

        return True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return False
    finally:
        node.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-pose face enrollment (10 frontal + 3 per angle)"
    )
    parser.add_argument('name', type=str, help='Name of person to enroll')
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    success = enroll_face_multipose(args.name, args.config)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
