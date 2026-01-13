# Quick Start Guide

## Setup

1. **Download models** (one-time):
   ```bash
   python3 scripts/download_models.py
   ```

2. **Configure authorized users** in `config/recognition_config.yaml`:
   ```yaml
   authorized_users:
     - evan_nicholas
   ```

## Enrollment (Multi-Pose for Best Accuracy)

Use the **multi-pose enrollment** for best results:

```bash
python3 scripts/enroll_face.py "evan_nicholas"
```

This will guide you through capturing **5 different poses**:
- **Frontal** (10 samples) - look straight at camera
- **Left** (3 samples) - turn head left
- **Right** (3 samples) - turn head right
- **Up** (3 samples) - tilt head up
- **Down** (3 samples) - tilt head down

**Total: 22 samples across 5 poses**

**Instructions during enrollment:**
1. Follow on-screen instructions for each pose
2. Press **SPACE** when ready for that pose
3. System auto-captures samples every 0.5 seconds
4. Move to next pose when current pose completes
5. Done! Saves to `data/faces/evan_nicholas/` (5 separate pose files)

## Testing Recognition

```bash
python3 scripts/test_recognition.py
```

This will:
- Load all authorized users from config
- Show live camera feed
- Display **GREEN box** when it recognizes you (similarity ≥ 0.35)
- Display **RED box** for unknown faces (similarity < 0.35)
- Show similarity score (0.0 to 1.0) - higher = more confident

**Press 'q' to quit**

## Running the Main System

```bash
python3 -m src.main
```

Controls:
- **q** or **ESC**: Quit
- **e**: Enter enrollment mode (interactive)

## Configuration

### Similarity Threshold

Edit `config/recognition_config.yaml`:

```yaml
recognition:
  similarity_threshold: 0.35  # Lower = more lenient, Higher = more strict
```

- **0.30-0.35**: Recommended (good balance)
- **0.40+**: Very strict (fewer false positives, more false negatives)
- **0.25-**: Lenient (more false positives, fewer false negatives)

### Temporal Filtering (Reduce Jitter & False Detections)

Control how stable faces must be before displaying:

```yaml
temporal:
  min_frames_to_show: 3     # Consecutive frames required before showing
  max_frames_missing: 5     # Keep face alive if temporarily lost
  iou_threshold: 0.3        # Face matching threshold across frames
```

**min_frames_to_show:**
- **1**: Immediate display (no filtering, more jitter)
- **3-5**: Recommended (reduces false detections)
- **10+**: Very stable (slow to respond)

**max_frames_missing:**
- **0**: Immediate removal when lost (more jitter)
- **5-10**: Recommended (smooth tracking)
- **20+**: Long persistence (may show stale faces)

**iou_threshold:**
- **0.2-0.4**: Recommended (tracks moving faces)
- **0.5+**: Strict matching (may create duplicate tracks)
- **0.1-**: Loose matching (may merge different faces)

### Adding More Users

1. Enroll the new user:
   ```bash
   python3 scripts/enroll_face.py "john_doe"
   ```

2. Add to authorized list in `config/recognition_config.yaml`:
   ```yaml
   authorized_users:
     - evan_nicholas
     - john_doe
   ```

3. Restart the system

## File Structure

```
data/faces/
├── evan_nicholas/
│   ├── frontal.npz   # Frontal pose embedding
│   ├── left.npz      # Left pose embedding
│   ├── right.npz     # Right pose embedding
│   ├── up.npz        # Up pose embedding
│   └── down.npz      # Down pose embedding
├── john_doe/
│   └── ...
```

Each user gets their own directory with 5 pose-specific embedding files

## Tips for Better Accuracy

1. **Use multi-pose enrollment** - captures 5 different angles (22 total samples)
2. **Good lighting** during enrollment (bright, even lighting)
3. **Follow pose instructions carefully** - turn/tilt head clearly for each pose
4. **Keep face in frame** during auto-capture sequence
5. **Adjust threshold** if too many false positives/negatives
6. **Re-enroll** if accuracy degrades (lighting changes, glasses, etc.)

## Troubleshooting

**Not recognizing you:**
- Lower the threshold (0.30 or 0.28)
- Re-enroll in similar lighting conditions to where you'll use the system
- Check lighting conditions

**Too many false positives:**
- Raise the threshold (0.38 or 0.40)
- Re-enroll with better quality samples
- Increase `min_frames_to_show` (5-7) to filter brief false detections

**Jittery/flickering boxes:**
- Increase `min_frames_to_show` (5-7)
- Increase `max_frames_missing` (10-15)
- Face should appear more stable

**Slow to respond/show faces:**
- Decrease `min_frames_to_show` (1-2)
- Decrease `max_frames_missing` (2-3)
- Trade-off: less stability for faster response

**Boxes disappear too quickly:**
- Increase `max_frames_missing` (10-20)
- Helps with brief occlusions (hand covering face, etc.)

**No faces detected:**
- Check camera is working
- Ensure good lighting
- Face should be clearly visible
