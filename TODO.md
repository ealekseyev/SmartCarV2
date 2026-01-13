# SmartCarV2 - Future Optimization Wishlist

## Phase 2: Performance Optimization (Post-MVP)

### TensorRT Optimization
- [ ] Convert BlazeFace ONNX model to TensorRT engine with FP16 precision
- [ ] Convert MobileFaceNet to TensorRT engine with FP16 precision
- [ ] Implement INT8 quantization with calibration dataset
- [ ] Create platform-aware model loader (TensorRT for Jetson, fallback for RPi)
- [ ] Add TensorRT engine caching for faster startup

### Platform-Specific Tuning
- [ ] Raspberry Pi: Explore ncnn or OpenVINO for CPU optimization
- [ ] Raspberry Pi: Implement model pruning for lower memory footprint
- [ ] Jetson Orin Nano: Leverage CUDA streams for parallel processing
- [ ] Jetson Orin Nano: Use NPP (NVIDIA Performance Primitives) for preprocessing
- [ ] Add power profiling and power-efficiency modes

### Pipeline Optimization
- [ ] Multi-threaded frame processing (separate threads for capture, detect, recognize)
- [ ] Implement frame skipping strategies for compute-constrained scenarios
- [ ] Add GPU-accelerated preprocessing (resize, normalization, color conversion)
- [ ] Batch processing for multiple faces when possible
- [ ] Smart frame buffering to handle FPS drops gracefully

### Model Improvements
- [ ] Implement face tracking to reduce redundant detections
- [ ] Add temporal smoothing for recognition confidence
- [ ] Explore model distillation for smaller MobileFaceNet variants
- [ ] Add liveness detection / anti-spoofing module
- [ ] Multi-scale face detection for varying distances

### Database & Storage
- [ ] Implement efficient embedding database (FAISS or similar)
- [ ] Add face gallery management (add/remove/update faces)
- [ ] Persistence layer for configuration and embeddings
- [ ] Support for multiple enrolled users

### Monitoring & Debugging
- [ ] Add performance profiling tools (layer-wise timing)
- [ ] Implement metrics collection (FPS, latency, accuracy)
- [ ] Remote monitoring dashboard
- [ ] Data collection mode for model improvement

### Hardware Integration
- [ ] CSI camera support (native MIPI for Jetson)
- [ ] Multiple camera support
- [ ] Hardware-accelerated video encoding for logging
- [ ] GPIO integration for car control signals

## Phase 3: Production Features

- [ ] Secure face database encryption
- [ ] Over-the-air model updates
- [ ] Fail-safe mechanisms and error recovery
- [ ] System health monitoring
- [ ] Auto-start service configuration
- [ ] Logging and audit trail
