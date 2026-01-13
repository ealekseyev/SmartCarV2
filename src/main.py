"""
Main node for SmartCar Face Recognition System.

Orchestrates and controls all subsystems:
- Camera subsystem
- Face detection and recognition pipeline
- Live viewer
"""

import argparse
import time
import yaml
from pathlib import Path
from collections import deque
from typing import Optional
from loguru import logger

from src.camera import CameraBase, USBCamera
from src.camera.camera_base import CameraSpecs
from src.pipeline import BlazeFaceDetector, MobileFaceNetRecognizer, FacePipeline
from src.pipeline.face_authenticator import FaceAuthenticator
from src.viewer import LiveViewer


class SmartCarNode:
    """
    Main node controlling the face recognition system.

    Manages lifecycle of all components and coordinates their operation.
    """

    def __init__(
        self,
        config_path: str = "config/camera_config.yaml",
        recognition_config_path: str = "config/recognition_config.yaml"
    ):
        """
        Initialize SmartCar node.

        Args:
            config_path: Path to camera configuration file
            recognition_config_path: Path to recognition configuration file
        """
        self.config_path = config_path
        self.recognition_config_path = recognition_config_path
        self.config = self._load_config()
        self.recognition_config = self._load_recognition_config()

        # Components
        self.camera: Optional[CameraBase] = None
        self.detector: Optional[BlazeFaceDetector] = None
        self.recognizer: Optional[MobileFaceNetRecognizer] = None
        self.pipeline: Optional[FacePipeline] = None
        self.viewer: Optional[LiveViewer] = None
        self.authenticator: Optional[FaceAuthenticator] = None

        # State
        self.running = False

        # State machine
        self.car_state = "LOCKED"
        self.recognized_frames = None  # Will be initialized in run()
        self.state_machine_enabled = False

        logger.info("SmartCar node initialized")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            logger.success(f"Loaded configuration from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _load_recognition_config(self) -> dict:
        """Load recognition configuration from YAML file."""
        try:
            config_file = Path(self.recognition_config_path)
            if not config_file.exists():
                logger.warning(f"Recognition config not found: {self.recognition_config_path}, using defaults")
                return self._get_default_recognition_config()

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            logger.success(f"Loaded recognition config from {self.recognition_config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading recognition config: {e}, using defaults")
            return self._get_default_recognition_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'camera': {
                'type': 'usb',
                'device_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'format': 'RGB'
            }
        }

    def _get_default_recognition_config(self) -> dict:
        """Get default recognition configuration."""
        return {
            'recognition': {
                'similarity_threshold': 0.35,
                'embedding_size': 512,
                'device': 'cpu',
                'faces_dir': 'data/faces'
            },
            'authorized_users': [],
            'detection': {
                'confidence_threshold': 0.7,
                'min_face_size': 40,
                'crop_padding': 0.2
            }
        }

    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing subsystems...")

            # Initialize camera
            if not self._init_camera():
                return False

            # Initialize detector
            if not self._init_detector():
                return False

            # Initialize recognizer
            if not self._init_recognizer():
                return False

            # Initialize pipeline
            if not self._init_pipeline():
                return False

            # Initialize viewer
            if not self._init_viewer():
                return False

            # Initialize authenticator
            if not self._init_authenticator():
                return False

            logger.success("All subsystems initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False

    def _init_camera(self) -> bool:
        """Initialize camera subsystem."""
        try:
            cam_config = self.config['camera']

            # Create camera specs
            specs = CameraSpecs(
                width=cam_config['width'],
                height=cam_config['height'],
                fps=cam_config['fps'],
                format=cam_config['format']
            )

            # Create camera instance based on type
            if cam_config['type'] == 'usb':
                self.camera = USBCamera(
                    specs=specs,
                    device_id=cam_config['device_id']
                )
            else:
                logger.error(f"Unsupported camera type: {cam_config['type']}")
                return False

            # Open camera
            if not self.camera.open():
                logger.error("Failed to open camera")
                return False

            logger.success("Camera initialized")
            return True

        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False

    def _init_detector(self) -> bool:
        """Initialize face detector."""
        try:
            self.detector = BlazeFaceDetector(
                confidence_threshold=0.7,
                use_opencv_dnn=True  # Using OpenCV fallback for now
            )

            logger.success("Detector initialized")
            return True

        except Exception as e:
            logger.error(f"Detector initialization error: {e}")
            return False

    def _init_recognizer(self) -> bool:
        """Initialize face recognizer."""
        try:
            rec_config = self.recognition_config.get('recognition', {})

            self.recognizer = MobileFaceNetRecognizer(
                embedding_size=rec_config.get('embedding_size', 512),
                similarity_threshold=rec_config.get('similarity_threshold', 0.35),
                device=rec_config.get('device', 'cpu')
            )

            logger.success("Recognizer initialized")

            # Load authorized users
            authorized_users = self.recognition_config.get('authorized_users', [])
            faces_dir = rec_config.get('faces_dir', 'data/faces')

            if authorized_users:
                loaded_count = self.recognizer.load_authorized_users(authorized_users, faces_dir)
                if loaded_count == 0:
                    logger.warning("No authorized users loaded! System will not recognize anyone.")
            else:
                logger.warning("No authorized users configured in recognition_config.yaml")

            return True

        except Exception as e:
            logger.error(f"Recognizer initialization error: {e}")
            return False

    def _init_pipeline(self) -> bool:
        """Initialize face pipeline."""
        try:
            # Get config values
            det_config = self.recognition_config.get('detection', {})
            temporal_config = self.recognition_config.get('temporal', {})

            self.pipeline = FacePipeline(
                detector=self.detector,
                recognizer=self.recognizer,
                min_face_size=det_config.get('min_face_size', 80),
                crop_padding=det_config.get('crop_padding', 0.2),
                enable_tracking=True,  # Enable temporal tracking
                min_frames_to_show=temporal_config.get('min_frames_to_show', 3),
                max_frames_missing=temporal_config.get('max_frames_missing', 5),
                iou_threshold=temporal_config.get('iou_threshold', 0.3)
            )

            logger.success("Pipeline initialized")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization error: {e}")
            return False

    def _init_viewer(self) -> bool:
        """Initialize live viewer."""
        try:
            self.viewer = LiveViewer(
                window_name="SmartCar Face Recognition",
                show_fps=True,
                show_confidence=True
            )

            logger.success("Viewer initialized")
            return True

        except Exception as e:
            logger.error(f"Viewer initialization error: {e}")
            return False

    def _init_authenticator(self) -> bool:
        """Initialize face authenticators for state machine (one per authorized user)."""
        try:
            # Check if state machine is enabled
            state_config = self.recognition_config.get('state_machine', {})
            self.state_machine_enabled = state_config.get('enabled', False)

            if not self.state_machine_enabled:
                logger.info("State machine disabled in config")
                return True

            # Get authentication config
            auth_config = self.recognition_config.get('authentication', {})

            # Support both single user (string) and multiple users (list)
            auth_users = auth_config.get('authorized_users', [])
            if isinstance(auth_users, str):
                auth_users = [auth_users]
            elif not auth_users:
                # Fallback to old 'authorized_user' config
                auth_users = [auth_config.get('authorized_user', 'evan_nicholas')]

            # Create authenticator for each user
            self.authenticator = {}
            for user in auth_users:
                self.authenticator[user] = FaceAuthenticator(
                    authorized_user=user,
                    min_frames_required=auth_config.get('min_frames_required', 5),
                    weighted_threshold=auth_config.get('weighted_confidence_threshold', 0.75),
                    min_single_frame=auth_config.get('min_single_frame_confidence', 0.60),
                    max_bbox_movement=auth_config.get('max_bbox_movement_pixels', 100.0),
                    history_size=auth_config.get('history_size', 10)
                )

            logger.success(f"Authenticator initialized for {len(self.authenticator)} user(s): {list(self.authenticator.keys())}")
            return True

        except Exception as e:
            logger.error(f"Authenticator initialization error: {e}")
            return False

    def run(self):
        """
        Main run loop.

        Captures frames, processes through pipeline, and displays results.
        """
        logger.info("Starting main run loop")
        self.running = True

        # Initialize state machine window if enabled
        if self.state_machine_enabled:
            state_config = self.recognition_config.get('state_machine', {})
            unlock_window_frames = state_config.get('unlock_window_frames', 500)
            unlock_required_detections = state_config.get('unlock_required_detections', 2)
            self.recognized_frames = deque(maxlen=unlock_window_frames)

            # Get authorized users list
            auth_config = self.recognition_config.get('authentication', {})
            auth_users = auth_config.get('authorized_users', [])
            if isinstance(auth_users, str):
                auth_users = [auth_users]
            elif not auth_users:
                auth_users = [auth_config.get('authorized_user', 'evan_nicholas')]

            logger.info(f"State machine enabled: window={unlock_window_frames}, required={unlock_required_detections}, users={auth_users}")
        else:
            unlock_window_frames = 500
            unlock_required_detections = 2
            auth_users = []

        # FPS calculation
        fps_start = time.time()
        frame_count = 0
        current_fps = 0.0

        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    continue

                # Process frame through pipeline
                results = self.pipeline.process_frame(frame)

                # State machine logic
                if self.state_machine_enabled:
                    # Find any authorized user in results
                    authorized_detected = any(r.identity in auth_users for r in results)

                    if self.car_state == "LOCKED":
                        # Only run authenticator if faces detected
                        if len(results) > 0:
                            # Debug logging
                            identities = [f"{r.identity}({r.similarity:.2f})" for r in results]
                            logger.debug(f"LOCKED - Detected: {identities}, Looking for any of: {auth_users}")

                            # Try to authenticate with any authorized user
                            authenticated = False
                            for result in results:
                                if result.identity in auth_users:
                                    # Update corresponding authenticator
                                    auth_result = self.authenticator[result.identity].update(result)
                                    logger.debug(f"Auth result for {result.identity}: {auth_result.reason} (conf={auth_result.confidence:.2f})")

                                    # Check for unlock
                                    if auth_result.authenticated:
                                        self.car_state = "UNLOCKED"
                                        self.recognized_frames.clear()
                                        logger.success(f"🔓 CAR UNLOCKED by {result.identity}")
                                        authenticated = True
                                        break

                            # If not authenticated, update all authenticators with None
                            if not authenticated:
                                for user in auth_users:
                                    if user not in [r.identity for r in results]:
                                        self.authenticator[user].update(None)

                    elif self.car_state == "UNLOCKED":
                        # Track authorized user detections (any confidence)
                        self.recognized_frames.append(1 if authorized_detected else 0)

                        # Check stay-unlocked criteria
                        detections_in_window = sum(self.recognized_frames)
                        if detections_in_window < unlock_required_detections:
                            self.car_state = "LOCKED"
                            # Reset all authenticators
                            for authenticator in self.authenticator.values():
                                authenticator.reset()
                            self.recognized_frames.clear()
                            logger.warning("🔒 CAR LOCKED")

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                # Display results
                extra_info = {
                    "Faces": len(results),
                    "DB": self.recognizer.get_database_size()
                }

                # Add state to display
                if self.state_machine_enabled:
                    extra_info["State"] = f"{'🔓 UNLOCKED' if self.car_state == 'UNLOCKED' else '🔒 LOCKED'}"

                self.viewer.show(frame, results, current_fps, extra_info)

                # Handle keyboard input
                key = self.viewer.wait_key(1)
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Quit requested")
                    break
                elif key == ord('e'):
                    logger.info("Enrollment mode - press 'c' to capture")
                    self._enrollment_mode()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.running = False
            logger.info("Main loop stopped")

    def _enrollment_mode(self):
        """
        Interactive enrollment mode.

        Allows user to enroll faces from live camera feed.
        """
        logger.info("Enrollment mode active - press 'c' to capture, 'q' to quit")

        while True:
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Detect faces for preview
            detections = self.detector.detect(frame)

            # Create temporary results for display
            from src.pipeline.face_pipeline import PipelineResult
            preview_results = []
            for det in detections:
                face_crop = self.detector.crop_face(frame, det)
                preview_results.append(
                    PipelineResult(
                        detection=det,
                        identity="[Press 'c' to enroll]",
                        similarity=0.0,
                        face_crop=face_crop
                    )
                )

            # Display
            self.viewer.show(frame, preview_results, 0.0, {"Mode": "Enrollment"})

            # Handle keys
            key = self.viewer.wait_key(1)

            if key == ord('c'):
                # Capture and enroll
                name = input("Enter name to enroll: ").strip()
                if name:
                    success = self.pipeline.enroll_from_frame(frame, name)
                    if success:
                        logger.success(f"Enrolled {name}")
                    else:
                        logger.error(f"Failed to enroll {name}")
                break

            elif key == ord('q') or key == 27:
                logger.info("Exiting enrollment mode")
                break

    def shutdown(self):
        """Shutdown all subsystems."""
        logger.info("Shutting down...")

        if self.viewer:
            self.viewer.destroy()

        if self.camera:
            self.camera.release()

        logger.success("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SmartCar Face Recognition System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to camera configuration file'
    )
    parser.add_argument(
        '--recognition-config',
        type=str,
        default='config/recognition_config.yaml',
        help='Path to recognition configuration file'
    )

    args = parser.parse_args()

    # Create and initialize node
    node = SmartCarNode(
        config_path=args.config,
        recognition_config_path=args.recognition_config
    )

    if not node.initialize():
        logger.error("Failed to initialize system")
        return 1

    # Run main loop
    try:
        node.run()
    finally:
        node.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
