#!/usr/bin/env python3
"""Live BlazeFace test - press SPACE to pause/detect, Q to quit."""

import cv2
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import create_face_detector


def estimate_head_pose(landmarks):
    """
    Estimate head pose from 5 facial landmarks.
    landmarks: [(x,y)] - [left_eye, right_eye, nose, left_mouth, right_mouth]
    Returns: (yaw, pitch) in degrees and pose_label
    """
    if not landmarks or len(landmarks) != 5:
        return 0, 0, "unknown"

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    # Calculate eye center
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2

    # Calculate mouth center
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2

    # Yaw (left-right): nose position relative to eye center
    eye_width = abs(right_eye[0] - left_eye[0])
    nose_offset = nose[0] - eye_center_x
    yaw = (nose_offset / eye_width) * 40 if eye_width > 0 else 0  # Scale to degrees

    # Pitch (up-down): vertical distance from eyes to mouth
    face_height = abs(mouth_center_y - eye_center_y)
    expected_height = eye_width * 1.2  # Rough ratio
    pitch = ((face_height - expected_height) / expected_height) * 30 if expected_height > 0 else 0

    # Determine pose label
    yaw_label = "RIGHT" if yaw > 10 else "LEFT" if yaw < -10 else "CENTER"
    pitch_label = "DOWN" if pitch > 8 else "UP" if pitch < -8 else ""

    if yaw_label == "CENTER" and not pitch_label:
        pose_label = "CENTER"
    elif pitch_label:
        pose_label = f"{pitch_label}-{yaw_label}"
    else:
        pose_label = yaw_label

    return yaw, pitch, pose_label


def align_face(image, landmarks, bbox, output_size=(256, 256)):
    """
    Align and frontalize face using perspective transformation.
    Handles rotation, translation, and out-of-plane rotations.
    """
    if not landmarks or len(landmarks) != 5:
        return None

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    # Standard frontal face template (based on average face proportions)
    # These are ideal positions for a perfectly frontal face
    ref_pts = np.array([
        [0.31, 0.36],   # left eye
        [0.69, 0.36],   # right eye
        [0.50, 0.55],   # nose tip
        [0.36, 0.75],   # left mouth corner
        [0.64, 0.75]    # right mouth corner
    ], dtype=np.float32) * output_size[0]

    # Source landmarks
    src_pts = np.array(landmarks, dtype=np.float32)

    # Compute similarity transform (handles rotation, scale, translation)
    M = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)[0]

    if M is None:
        return None

    # Apply transformation to align face to frontal template
    aligned = cv2.warpAffine(image, M, output_size,
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    return aligned


# Create detector
detector = create_face_detector('blazeface', model_path='models/insightface/det_10g.onnx', confidence_threshold=0.5)

# Open camera
cap = cv2.VideoCapture(0)
paused = False
frozen_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not paused:
        display = frame.copy()
    else:
        display = frozen_frame.copy() if frozen_frame is not None else frame.copy()

    # Always detect when paused
    if paused:
        rgb = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect(rgb)

        overlay_y = 10
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{det.confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw landmarks
            if det.landmarks:
                for lm in det.landmarks:
                    cv2.circle(display, lm, 3, (0, 0, 255), -1)

                # Estimate head pose
                yaw, pitch, pose_label = estimate_head_pose(det.landmarks)

                # Display pose
                pose_y = y2 + 25
                cv2.putText(display, f"Pose: {pose_label}", (x1, pose_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(display, f"Y:{yaw:.1f} P:{pitch:.1f}", (x1, pose_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Align face and show as overlay
                aligned = align_face(frozen_frame, det.landmarks, det.bbox, output_size=(256, 256))
                if aligned is not None:
                    # Position overlay in top-right corner
                    h, w = aligned.shape[:2]
                    overlay_x = display.shape[1] - w - 10

                    # Add border
                    cv2.rectangle(display, (overlay_x-2, overlay_y-2),
                                (overlay_x+w+2, overlay_y+h+2), (255, 255, 255), 2)

                    # Overlay aligned face
                    display[overlay_y:overlay_y+h, overlay_x:overlay_x+w] = aligned

                    # Label
                    cv2.putText(display, "Aligned", (overlay_x, overlay_y-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    overlay_y += h + 20

        cv2.putText(display, f"Faces: {len(dets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('BlazeFace Test', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        if paused:
            frozen_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()
