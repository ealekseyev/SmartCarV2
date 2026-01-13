# Model Setup Guide

This guide explains how to download and integrate pre-trained models for BlazeFace and MobileFaceNet.

## Quick Start

The system currently uses placeholder/fallback implementations. To use actual models:

1. Download pre-trained models (see below)
2. Place in `models/` directory
3. Update code to load models (marked with TODO comments)

## BlazeFace Face Detection

### Option 1: MediaPipe BlazeFace (Recommended)

```bash
# Install MediaPipe
pip install mediapipe

# MediaPipe includes BlazeFace - no separate download needed
```

**Integration**: Modify `src/pipeline/face_detector.py` to use MediaPipe's face detection.

### Option 2: ONNX BlazeFace

Download from PINTO model zoo or convert from TensorFlow:

```bash
# Clone PINTO model zoo
git clone https://github.com/PINTO0309/PINTO_model_zoo
cd PINTO_model_zoo/033_Face_Detection_BlazeFace

# Copy ONNX model to your project
cp blazeface.onnx /path/to/SmartCarV2/models/blazeface/
```

**Integration**: Update `src/pipeline/face_detector.py` to load ONNX model with onnxruntime.

### Option 3: Keep Haar Cascade (Current)

The system currently uses OpenCV's Haar Cascade as a fallback. While not as accurate as BlazeFace, it works without additional models.

## MobileFaceNet Face Recognition

### Option 1: InsightFace Pre-trained Models (Recommended)

```bash
# Install insightface
pip install insightface

# Or download models manually from:
# https://github.com/deepinsight/insightface/tree/master/model_zoo
```

**Models to download**:
- `mobilefacenet_v1.onnx` or `mobilefacenet_v2.onnx`
- Trained with ArcFace loss on MS1MV2 dataset

**Place in**: `models/mobilefacenet/`

**Integration Steps**:

1. Download model:
```bash
# Example using model from InsightFace
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip -d models/insightface/
```

2. Update `src/pipeline/face_recognizer.py`:

```python
import onnxruntime as ort

def load_model(self, model_path: str) -> bool:
    try:
        self.model = ort.InferenceSession(model_path)
        self._is_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
    if not self._is_loaded:
        return None

    # Preprocess
    input_tensor = self.preprocess_face(face_image)

    # Run inference
    onnx_inputs = {self.model.get_inputs()[0].name: input_tensor.cpu().numpy()}
    embedding = self.model.run(None, onnx_inputs)[0]

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.flatten()
```

### Option 2: PyTorch Pre-trained Weights

From [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch):

```bash
# Clone repository
git clone https://github.com/cavalleria/cavaface.pytorch
cd cavaface.pytorch

# Download pre-trained weights
# Follow their instructions to download MobileFaceNet weights
# Usually available on Google Drive or Baidu Drive

# Copy to your project
cp mobilefacenet_model_best.pth /path/to/SmartCarV2/models/mobilefacenet/
```

**Integration**: Load PyTorch model in `src/pipeline/face_recognizer.py`:

```python
from mobilefacenet import MobileFaceNet  # From cavaface repo

def load_model(self, model_path: str) -> bool:
    self.model = MobileFaceNet(embedding_size=512)
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()
    self._is_loaded = True
    return True
```

### Option 3: Convert to ONNX (For Deployment)

If you have PyTorch weights, convert to ONNX for better portability:

```python
import torch
from mobilefacenet import MobileFaceNet

# Load PyTorch model
model = MobileFaceNet(embedding_size=512)
model.load_state_dict(torch.load('mobilefacenet.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 112, 112)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'models/mobilefacenet/mobilefacenet.onnx',
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
)
```

## Model Directory Structure

After setup, your `models/` directory should look like:

```
models/
├── blazeface/
│   ├── blazeface.onnx
│   └── anchors.npy (if needed)
└── mobilefacenet/
    ├── mobilefacenet.onnx (or .pth)
    └── model_info.txt
```

## TensorRT Conversion (Jetson Orin Nano)

Once you have ONNX models, convert to TensorRT for optimized inference:

```bash
# Convert BlazeFace
trtexec --onnx=models/blazeface/blazeface.onnx \
        --saveEngine=models/tensorrt/blazeface_fp16.trt \
        --fp16

# Convert MobileFaceNet
trtexec --onnx=models/mobilefacenet/mobilefacenet.onnx \
        --saveEngine=models/tensorrt/mobilefacenet_fp16.trt \
        --fp16
```

**INT8 Quantization** (requires calibration dataset):

```bash
trtexec --onnx=models/mobilefacenet/mobilefacenet.onnx \
        --saveEngine=models/tensorrt/mobilefacenet_int8.trt \
        --int8 \
        --calib=/path/to/calibration_data
```

See `TODO.md` for full TensorRT integration plans.

## Recommended Setup

For **development/testing**:
- BlazeFace: Use MediaPipe (easiest, no model download)
- MobileFaceNet: Keep placeholder or use ONNX from InsightFace

For **Raspberry Pi deployment**:
- BlazeFace: ONNX with ONNX Runtime
- MobileFaceNet: ONNX with ONNX Runtime
- Consider model quantization for faster inference

For **Jetson Orin Nano deployment**:
- BlazeFace: TensorRT FP16 engine
- MobileFaceNet: TensorRT FP16/INT8 engine
- Use CUDA/cuDNN for preprocessing

## Testing Models

After integrating models, test with:

```bash
# Run system with debug logging
python -m src.main

# Check if models loaded successfully (look for log messages)
# "BlazeFace model loaded"
# "MobileFaceNet model loaded"

# Test face detection and recognition
# Enroll a face and verify it's recognized
```

## Resources

- **InsightFace**: https://github.com/deepinsight/insightface
- **MediaPipe**: https://google.github.io/mediapipe/solutions/face_detection
- **PINTO Model Zoo**: https://github.com/PINTO0309/PINTO_model_zoo
- **cavaface.pytorch**: https://github.com/cavalleria/cavaface.pytorch
- **TensorRT**: https://docs.nvidia.com/deeplearning/tensorrt/

## Troubleshooting

**Model not loading**:
- Check file paths in code
- Verify model file integrity
- Check ONNX Runtime or PyTorch installation

**Low accuracy**:
- Ensure proper preprocessing (normalization, resizing)
- Check input size matches model expectation (usually 112x112 for MobileFaceNet)
- Verify model is trained with ArcFace loss

**Slow inference**:
- Use ONNX instead of PyTorch for deployment
- Enable TensorRT on Jetson
- Reduce resolution or use FP16/INT8 quantization
