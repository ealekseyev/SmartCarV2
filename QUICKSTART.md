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

## Enrollment (Better Accuracy)

Use the **advanced enrollment** for best results:

```bash
python3 scripts/enroll_face_advanced.py "evan_nicholas"
```

This will:
- Capture **10 samples** of your face (adjustable with `--samples`)
- Average the embeddings for robustness
- Check quality/consistency
- Save to `data/faces/evan_nicholas.npz`

**Instructions during enrollment:**
1. Position your face in camera view
2. Press **SPACE** to start
3. Slowly move your head (left, right, up, down)
4. System auto-captures samples every 0.5 seconds
5. Done! File saved automatically

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

### Adding More Users

1. Enroll the new user:
   ```bash
   python3 scripts/enroll_face_advanced.py "john_doe"
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
├── evan_nicholas.npz    # Your face embedding
├── john_doe.npz         # Another user's embedding
└── ...
```

Each user gets their own file: `{username}.npz`

## Tips for Better Accuracy

1. **Use advanced enrollment** with 10+ samples
2. **Good lighting** during enrollment and recognition
3. **Face the camera directly** during enrollment
4. **Adjust threshold** if too many false positives/negatives
5. **Re-enroll** if accuracy degrades (lighting changes, glasses, etc.)

## Troubleshooting

**Not recognizing you:**
- Lower the threshold (0.30 or 0.28)
- Re-enroll with more samples (--samples 15)
- Check lighting conditions

**Too many false positives:**
- Raise the threshold (0.38 or 0.40)
- Re-enroll with better quality samples

**No faces detected:**
- Check camera is working
- Ensure good lighting
- Face should be clearly visible
