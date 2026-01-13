#!/usr/bin/env python3
"""Test BlazeFace/SCRFD detector."""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import create_face_detector


def main():
    """Test BlazeFace detector."""
    print("Testing BlazeFace/SCRFD Detector")
    print("=" * 60)

    # Create detector
    print("\n1. Creating BlazeFace detector...")
    detector = create_face_detector(
        detector_type='blazeface',
        model_path='models/insightface/det_10g.onnx',
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    if not detector.is_loaded():
        print("✗ Failed to load detector")
        return 1

    print("✓ Detector loaded successfully")

    # Test on random image
    print("\n2. Testing on random image (640x480)...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        detections = detector.detect(test_image)
        print(f"✓ Detection successful: {len(detections)} faces found")
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test on actual image if available
    print("\n3. Looking for test images...")
    test_images = [
        "test.jpg",
        "test.png",
        "data/test.jpg",
        "data/test.png"
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n4. Testing on {img_path}...")
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                detections = detector.detect(img_rgb)
                print(f"✓ Detected {len(detections)} faces")

                for i, det in enumerate(detections):
                    print(f"   Face {i+1}: bbox={det.bbox}, conf={det.confidence:.3f}, landmarks={len(det.landmarks) if det.landmarks else 0}")

                    # Draw on image
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{det.confidence:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw landmarks
                    if det.landmarks:
                        for lm in det.landmarks:
                            cv2.circle(img, lm, 2, (0, 0, 255), -1)

                # Save result
                output_path = f"blazeface_result_{Path(img_path).stem}.jpg"
                cv2.imwrite(output_path, img)
                print(f"✓ Saved result to {output_path}")
            break

    print("\n" + "=" * 60)
    print("✓ BlazeFace detector test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
