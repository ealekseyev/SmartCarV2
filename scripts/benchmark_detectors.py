#!/usr/bin/env python3
"""
Benchmark different face detectors to compare performance.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from src.pipeline import create_face_detector


def benchmark_detector(detector, image, num_iterations=100):
    """Benchmark detector on image."""
    times = []

    # Warmup
    for _ in range(5):
        _ = detector.detect(image)

    # Benchmark
    for _ in range(num_iterations):
        start = time.perf_counter()
        detections = detector.detect(image)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    }


def main():
    """Run benchmarks."""
    print("Face Detector Benchmark")
    print("=" * 60)

    # Create test images
    print("\nGenerating test images...")
    images = {
        '640x480': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        '1280x720': np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        '1920x1080': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
    }

    # Detectors to benchmark
    detector_types = ['haar_cascade', 'int8_haar_cascade']

    # Run benchmarks
    for img_name, image in images.items():
        print(f"\n{'=' * 60}")
        print(f"Resolution: {img_name}")
        print(f"{'=' * 60}")

        for detector_type in detector_types:
            print(f"\n{detector_type}:")
            try:
                detector = create_face_detector(detector_type)
                if not detector.is_loaded():
                    print("  ✗ Failed to load")
                    continue

                results = benchmark_detector(detector, image, num_iterations=50)

                print(f"  Mean:   {results['mean']:.2f} ms")
                print(f"  Median: {results['median']:.2f} ms")
                print(f"  Std:    {results['std']:.2f} ms")
                print(f"  Min:    {results['min']:.2f} ms")
                print(f"  Max:    {results['max']:.2f} ms")

            except Exception as e:
                print(f"  ✗ Error: {e}")

    print(f"\n{'=' * 60}")
    print("Benchmark complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
