"""
MobileFaceNet face recognition module with ArcFace.

Generates face embeddings using MobileFaceNet and performs face verification
using cosine similarity.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
from pathlib import Path
from loguru import logger
import onnxruntime as ort


class MobileFaceNetRecognizer:
    """
    Face recognition using MobileFaceNet with ArcFace.

    Generates 128-dim (or 512-dim) face embeddings for face verification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_size: int = 512,
        similarity_threshold: float = 0.6,
        device: str = "cpu"
    ):
        """
        Initialize MobileFaceNet recognizer.

        Args:
            model_path: Path to pre-trained MobileFaceNet model
            embedding_size: Size of face embedding (128 or 512)
            similarity_threshold: Cosine similarity threshold for verification
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_path = model_path
        self.embedding_size = embedding_size
        self.similarity_threshold = similarity_threshold

        # Set device
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)

        self.model = None
        self._is_loaded = False
        self.use_onnx = False

        # Face database: {name: embedding}
        self.face_database: Dict[str, np.ndarray] = {}

        # Input size for MobileFaceNet (typically 112x112)
        self.input_size = (112, 112)

        # Try to auto-load model if available
        self._auto_load_model()

    def _auto_load_model(self):
        """Automatically load model if found in standard locations."""
        search_paths = [
            "models/insightface/w600k_r50.onnx",  # InsightFace R50 (direct extract)
            "models/insightface/buffalo_l/w600k_r50.onnx",  # InsightFace R50 (in subdir)
            "models/insightface/w600k_mbf.onnx",  # InsightFace MobileFaceNet
            "models/mobilefacenet/mobilefacenet.onnx",
            "models/insightface/*/w600k_r50.onnx",  # Glob pattern
        ]

        for pattern in search_paths:
            if '*' in pattern:
                # Glob pattern
                from glob import glob
                matches = glob(pattern)
                if matches:
                    model_path = matches[0]
                    logger.info(f"Found model at {model_path}")
                    if self.load_model(model_path):
                        return
            else:
                # Direct path
                if Path(pattern).exists():
                    logger.info(f"Found model at {pattern}")
                    if self.load_model(pattern):
                        return

        logger.warning("No pre-trained model found. Run: python scripts/download_models.py")
        logger.warning("Will use placeholder embeddings (not for production!)")

    def load_model(self, model_path: str) -> bool:
        """
        Load pre-trained MobileFaceNet model.

        Args:
            model_path: Path to ONNX model

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info(f"Loading face recognition model from {model_path}")

            # Load ONNX model
            self.model = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']  # Use CPU for now
            )

            # Get input/output names
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # Get input shape
            input_shape = self.model.get_inputs()[0].shape
            logger.info(f"Model input shape: {input_shape}")

            # Update input size based on model
            if len(input_shape) == 4:  # (batch, channels, height, width)
                self.input_size = (input_shape[2], input_shape[3])

            self.model_path = model_path
            self._is_loaded = True
            self.use_onnx = True
            logger.success(f"Model loaded successfully. Input size: {self.input_size}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Continuing without model (placeholder embeddings only)")
            return False

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for MobileFaceNet.

        Args:
            face_image: Face crop in RGB format

        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)

        # Normalize to [-1, 1] (common for face recognition models)
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 127.5

        # Convert to tensor (C, H, W)
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)

        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)

        return face_tensor.to(self.device)

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image.

        Args:
            face_image: Cropped face image (RGB)

        Returns:
            Face embedding as numpy array, or None if failed
        """
        try:
            if self._is_loaded and self.use_onnx:
                # Use actual ONNX model
                return self._get_embedding_onnx(face_image)
            else:
                # Fallback: placeholder (not for production!)
                logger.warning("Model not loaded, using placeholder embedding")
                embedding = np.random.randn(self.embedding_size).astype(np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _get_embedding_onnx(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using ONNX model."""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_image, self.input_size)

            # Normalize: convert to float and normalize to [-1, 1] or [0, 1]
            # InsightFace models typically use [0, 1] normalization
            face_normalized = face_resized.astype(np.float32) / 255.0

            # Convert RGB to BGR (InsightFace models expect BGR)
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_RGB2BGR)

            # Transpose to (C, H, W) and add batch dimension
            face_input = np.transpose(face_normalized, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0)

            # Run inference
            outputs = self.model.run([self.output_name], {self.input_name: face_input})
            embedding = outputs[0].flatten()

            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            logger.debug(f"Generated embedding with shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Error in ONNX inference: {e}")
            return None

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Cosine similarity score [0, 1]
        """
        # Ensure normalized
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def verify(self, face_image: np.ndarray, name: str) -> tuple[bool, float]:
        """
        Verify if face matches enrolled person.

        Args:
            face_image: Cropped face image
            name: Name of person to verify against

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if name not in self.face_database:
            logger.warning(f"Person '{name}' not found in database")
            return False, 0.0

        # Get embedding for input face
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return False, 0.0

        # Compare with enrolled embedding
        enrolled_embedding = self.face_database[name]
        similarity = self.cosine_similarity(embedding, enrolled_embedding)

        is_match = similarity >= self.similarity_threshold

        logger.debug(f"Verification: {name}, similarity={similarity:.3f}, match={is_match}")
        return is_match, similarity

    def identify(self, face_image: np.ndarray) -> tuple[Optional[str], float]:
        """
        Identify face from enrolled database (checks all poses).

        Args:
            face_image: Cropped face image

        Returns:
            Tuple of (name, similarity_score) or (None, 0.0) if no match
        """
        if len(self.face_database) == 0:
            logger.warning("Face database is empty")
            return None, 0.0

        # Get embedding
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return None, 0.0

        # Find best match across all users and all their pose embeddings
        best_name = None
        best_similarity = 0.0
        best_pose = None

        # Pose priority: check frontal first, then others
        pose_order = ['frontal', 'left', 'right', 'up', 'down']

        for name, pose_embeddings in self.face_database.items():
            # pose_embeddings is now a dict: {'frontal': emb, 'left': emb, ...}
            if isinstance(pose_embeddings, dict):
                # New multi-pose format
                for pose in pose_order:
                    if pose in pose_embeddings:
                        enrolled_embedding = pose_embeddings[pose]
                        similarity = self.cosine_similarity(embedding, enrolled_embedding)

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_name = name
                            best_pose = pose
            else:
                # Legacy format (single embedding or array)
                # Handle for backward compatibility
                if pose_embeddings.ndim == 1:
                    embeddings_to_check = [pose_embeddings]
                else:
                    embeddings_to_check = pose_embeddings

                for enrolled_embedding in embeddings_to_check:
                    similarity = self.cosine_similarity(embedding, enrolled_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_name = name
                        best_pose = "legacy"

        # Check if above threshold
        if best_similarity >= self.similarity_threshold:
            logger.info(f"Identified: {best_name} (similarity={best_similarity:.3f}, pose={best_pose})")
            return best_name, best_similarity
        else:
            logger.debug(f"No match found (best similarity={best_similarity:.3f})")
            return None, best_similarity

    def enroll(self, name: str, face_image: np.ndarray) -> bool:
        """
        Enroll a person's face into the database.

        Args:
            name: Person's name
            face_image: Cropped face image

        Returns:
            bool: True if enrollment successful
        """
        embedding = self.get_embedding(face_image)

        if embedding is None:
            logger.error(f"Failed to enroll {name}: could not generate embedding")
            return False

        self.face_database[name] = embedding
        logger.success(f"Enrolled {name} into face database")
        return True

    def remove_person(self, name: str) -> bool:
        """Remove person from database."""
        if name in self.face_database:
            del self.face_database[name]
            logger.info(f"Removed {name} from database")
            return True
        return False

    def save_user_embedding(self, name: str, embedding: np.ndarray, pose: str = "frontal", faces_dir: str = "data/faces") -> bool:
        """
        Save individual user embedding to file (one file per pose).

        Args:
            name: Username
            embedding: Face embedding (single averaged embedding)
            pose: Pose name (frontal, left, right, up, down)
            faces_dir: Directory to store face files

        Returns:
            bool: True if saved successfully
        """
        try:
            # Create user directory
            user_dir = Path(faces_dir) / name
            user_dir.mkdir(parents=True, exist_ok=True)

            # Save pose-specific file
            filepath = user_dir / f"{pose}.npz"

            np.savez(filepath, embedding=embedding)
            logger.success(f"Saved {name}'s {pose} embedding to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving user embedding: {e}")
            return False

    def load_user_embedding(self, name: str, faces_dir: str = "data/faces") -> bool:
        """
        Load all pose embeddings for a user.

        Args:
            name: Username
            faces_dir: Directory where face files are stored

        Returns:
            bool: True if at least one embedding loaded successfully
        """
        try:
            user_dir = Path(faces_dir) / name

            if not user_dir.exists():
                logger.warning(f"No directory found for {name} at {user_dir}")
                return False

            # Load all pose files
            poses = ['frontal', 'left', 'right', 'up', 'down']
            embeddings = {}
            loaded_count = 0

            for pose in poses:
                pose_file = user_dir / f"{pose}.npz"

                if pose_file.exists():
                    try:
                        data = np.load(pose_file)
                        if 'embedding' in data:
                            embeddings[pose] = data['embedding']
                            loaded_count += 1
                            logger.debug(f"Loaded {pose} embedding for {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {pose} for {name}: {e}")

            if loaded_count == 0:
                logger.error(f"No valid embeddings found for {name}")
                return False

            # Store as dict of pose embeddings
            self.face_database[name] = embeddings
            logger.success(f"Loaded {name}'s embeddings ({loaded_count} poses)")
            return True

        except Exception as e:
            logger.error(f"Error loading user embeddings: {e}")
            return False

    def load_authorized_users(self, authorized_users: list, faces_dir: str = "data/faces") -> int:
        """
        Load embeddings for all authorized users.

        Args:
            authorized_users: List of authorized usernames
            faces_dir: Directory where face files are stored

        Returns:
            int: Number of users successfully loaded
        """
        loaded_count = 0

        logger.info(f"Loading {len(authorized_users)} authorized user(s)...")

        for username in authorized_users:
            if self.load_user_embedding(username, faces_dir):
                loaded_count += 1

        if loaded_count == 0:
            logger.warning("No authorized users loaded!")
        else:
            logger.success(f"Loaded {loaded_count}/{len(authorized_users)} authorized user(s)")

        return loaded_count

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def get_database_size(self) -> int:
        """Get number of enrolled faces."""
        return len(self.face_database)


# TODO: Implement actual MobileFaceNet model architecture or load pre-trained weights
# Resources:
# - https://github.com/deepinsight/insightface (pre-trained models)
# - https://github.com/cavalleria/cavaface.pytorch (MobileFaceNet + ArcFace implementation)
# - Convert to ONNX for deployment
