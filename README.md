# SmartCarV2 - Facial Recognition Auto-Start System

An intelligent facial recognition system designed to automatically start your car when it detects your face. Built with a modular architecture for easy deployment on Raspberry Pi and Jetson Orin Nano.

## Features

- **Modular Camera Subsystem**: Easily swap camera hardware with guaranteed resolution/FPS output (1280x720@30fps)
- **Fast Face Detection**: BlazeFace detector optimized for mobile/edge devices
- **Accurate Face Recognition**: MobileFaceNet with ArcFace for robust face verification
- **Live Preview**: Real-time display with bounding boxes, identity labels, and confidence scores
- **Easy Face Enrollment**: Simple tools to register authorized faces
- **Platform Support**: Raspberry Pi (initial) and Jetson Orin Nano (with TensorRT optimization)

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Camera    │────▶│   Pipeline   │────▶│   Viewer   │
│  Subsystem  │     │  (Det + Rec) │     │  (Display) │
└─────────────┘     └──────────────┘     └────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Main Node  │
                    └─────────────┘
```

### Components

1. **Camera Subsystem** (`src/camera/`)
   - `camera_base.py`: Abstract base class defining camera interface contract
   - `usb_camera.py`: USB camera implementation with guaranteed specs
   - Modular design for easy camera swapping

2. **Pipeline** (`src/pipeline/`)
   - `face_detector.py`: BlazeFace face detection
   - `face_recognizer.py`: MobileFaceNet + ArcFace recognition
   - `face_pipeline.py`: Orchestrates detection → recognition flow

3. **Viewer** (`src/viewer/`)
   - `live_viewer.py`: Real-time display with overlays

4. **Main Node** (`src/main.py`)
   - Initializes and controls all subsystems
   - Main run loop and event handling

## Installation

### Prerequisites

- Python 3.8+ (Raspberry Pi) or 3.10+ (Jetson)
- USB RGB camera
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SmartCarV2.git
cd SmartCarV2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Platform-Specific Setup

**Raspberry Pi:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-opencv libopencv-dev

# Run setup (future)
# bash scripts/setup_raspberry_pi.sh
```

**Jetson Orin Nano:**
```bash
# Install JetPack SDK with TensorRT support
# Follow NVIDIA's installation guide

# Install PyTorch for Jetson
# pip install torch torchvision (specific Jetson wheel)

# Run setup (future)
# bash scripts/setup_jetson.sh
```

## Quick Start

### 1. Configure Camera

Edit `config/camera_config.yaml` to match your camera:

```yaml
camera:
  type: usb
  device_id: 0  # /dev/video0
  width: 1280
  height: 720
  fps: 30
  format: RGB
```

### 2. Enroll Your Face

Use the enrollment script to register your face:

```bash
python scripts/enroll_face.py "YourName"
```

- Position your face in the camera view
- Press 'c' to capture and enroll
- Press 'q' to cancel

Your face will be saved to `data/face_database.npz`.

### 3. Run the System

```bash
python -m src.main
```

**Keyboard Controls:**
- `q` or `ESC`: Quit
- `e`: Enter enrollment mode

## Usage

### Basic Operation

```bash
# Run with default config
python -m src.main

# Run with custom config
python -m src.main --config path/to/config.yaml

# Load existing face database
python -m src.main --load-db data/face_database.npz

# Save database on exit
python -m src.main --save-db data/face_database.npz
```

### Face Enrollment

```bash
# Interactive enrollment
python scripts/enroll_face.py "PersonName"

# During runtime, press 'e' to enter enrollment mode
```

### Camera Specifications

The camera subsystem guarantees:
- **Resolution**: 1280x720 (configurable)
- **Frame Rate**: 30 FPS (configurable)
- **Format**: RGB (configurable to BGR)
- **Thread-safe**: Safe concurrent frame access

To swap cameras, create a new class inheriting from `CameraBase` and implement the required methods.

## Project Structure

```
SmartCarV2/
├── src/
│   ├── camera/          # Camera subsystem
│   ├── pipeline/        # Detection & recognition
│   ├── viewer/          # Live display
│   └── main.py          # Main orchestration node
├── config/              # Configuration files
├── scripts/             # Utility scripts
├── models/              # Model weights (not committed)
├── data/                # Face database (not committed)
├── requirements.txt     # Python dependencies
├── TODO.md             # Future optimization wishlist
└── README.md           # This file
```

## Development Status

### Current Implementation (MVP)

- [x] Modular camera subsystem with USB support
- [x] Face detection (using Haar Cascade as temporary fallback)
- [x] Face recognition framework (MobileFaceNet architecture ready)
- [x] Pipeline orchestration
- [x] Live viewer with overlays
- [x] Face enrollment tools
- [x] Main control node

### Next Steps (See TODO.md for full list)

- [ ] Integrate actual BlazeFace ONNX model
- [ ] Load pre-trained MobileFaceNet weights
- [ ] TensorRT optimization for Jetson
- [ ] Multi-threading for better FPS
- [ ] Liveness detection / anti-spoofing
- [ ] Car control integration

## Model Details

### Face Detection: BlazeFace

- Lightweight detector optimized for mobile
- Currently using OpenCV Haar Cascade as placeholder
- **TODO**: Integrate actual BlazeFace ONNX model

### Face Recognition: MobileFaceNet + ArcFace

- Embedding size: 512 dimensions
- Similarity threshold: 0.6 (cosine similarity)
- Currently uses placeholder random embeddings
- **TODO**: Load pre-trained weights from InsightFace or similar

## Performance

### Target Metrics

- **FPS**: 30 FPS (Jetson with TensorRT), 10-15 FPS (Raspberry Pi CPU)
- **Latency**: < 100ms detection + recognition
- **Accuracy**: > 95% face verification (with proper models)

### Current Status

- Basic pipeline functional with placeholder models
- FPS depends on camera and detection speed
- Ready for pre-trained model integration

## Troubleshooting

### Camera Issues

```bash
# List available cameras (Linux)
v4l2-ctl --list-devices

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Common Issues

1. **Camera not opening**: Check device_id in config, ensure camera is not used by another process
2. **Low FPS**: Reduce resolution or disable recognition for testing
3. **No faces detected**: Ensure good lighting and face is clearly visible

## Contributing

Contributions welcome! Areas of interest:

- Integrate pre-trained BlazeFace and MobileFaceNet models
- TensorRT optimization pipeline
- Additional camera implementations (CSI, MIPI)
- Liveness detection
- Performance benchmarks

## License

[Add your license here]

## Acknowledgments

- BlazeFace: MediaPipe
- MobileFaceNet: [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch)
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)

## Safety Notice

This system is intended for personal use on your own vehicle. Ensure compliance with local regulations regarding vehicle modifications and automated systems.
