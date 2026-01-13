# Face Detector Configuration Guide

The SmartCarV2 pipeline now supports multiple face detection backends that can be switched via configuration.

## Available Detectors

### 1. Haar Cascade (Default)
- **Type**: `haar_cascade`
- **Speed**: Fast (CPU-optimized)
- **Accuracy**: Low-Medium
- **Requirements**: None (included with OpenCV)
- **Best for**: CPU-only environments, quick prototyping

### 2. BlazeFace
- **Type**: `blazeface`
- **Speed**: Fast
- **Accuracy**: Medium-High
- **Requirements**: ONNX model file, onnxruntime
- **Best for**: Mobile/edge devices, balanced performance
- **Note**: Postprocessing not fully implemented yet

### 3. OpenCV DNN
- **Type**: `dnn`
- **Speed**: Medium
- **Accuracy**: High
- **Requirements**: ResNet-SSD model files
- **Best for**: Better accuracy on CPU

## Configuration

Edit `config/recognition_config.yaml`:

```yaml
detection:
  # Switch detector type here
  detector_type: haar_cascade  # Options: haar_cascade, blazeface, dnn

  # BlazeFace model path (only used if detector_type=blazeface)
  blazeface_model_path: "models/insightface/det_10g.onnx"

  confidence_threshold: 0.7
  nms_threshold: 0.3
```

## Usage Examples

### Example 1: Default Haar Cascade
```yaml
detection:
  detector_type: haar_cascade
  confidence_threshold: 0.7
```

### Example 2: BlazeFace with Auto-Discovery
```yaml
detection:
  detector_type: blazeface
  blazeface_model_path: ""  # Auto-search in models/
  confidence_threshold: 0.7
  nms_threshold: 0.3
```

### Example 3: Specific BlazeFace Model
```yaml
detection:
  detector_type: blazeface
  blazeface_model_path: "models/insightface/det_10g.onnx"
  confidence_threshold: 0.7
```

## Programmatic Usage

```python
from src.pipeline import create_face_detector

# Create Haar Cascade detector
detector = create_face_detector(
    detector_type='haar_cascade',
    confidence_threshold=0.7
)

# Create BlazeFace detector
detector = create_face_detector(
    detector_type='blazeface',
    model_path='models/insightface/det_10g.onnx',
    confidence_threshold=0.7,
    nms_threshold=0.3
)

# Use detector
import cv2
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = detector.detect(image_rgb)
```

## Performance Comparison

| Detector      | Speed (CPU) | Accuracy | Model Size | Memory  |
|---------------|-------------|----------|------------|---------|
| Haar Cascade  | ~50ms       | Low      | Included   | Low     |
| BlazeFace     | ~30ms       | Medium   | ~16MB      | Medium  |
| OpenCV DNN    | ~70ms       | High     | ~10MB      | Medium  |

*Benchmarks approximate on typical laptop CPU

## Adding New Detectors

To add a new detector backend:

1. Create class inheriting from `FaceDetectorBase` in `src/pipeline/face_detector.py`
2. Implement required methods: `detect()`, `is_loaded()`
3. Add case to `create_face_detector()` factory function
4. Update this documentation

## Troubleshooting

**BlazeFace not detecting faces:**
- Ensure model postprocessing is implemented for your specific model
- Check model input/output shapes match expectations
- Verify onnxruntime is installed: `pip install onnxruntime`

**DNN detector fails to load:**
- Download model files from OpenCV repository
- Place in `models/opencv/` directory

**Haar Cascade too many false positives:**
- Increase `min_face_size` in config
- Adjust temporal filtering parameters
