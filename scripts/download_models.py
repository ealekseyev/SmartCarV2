#!/usr/bin/env python3
"""
Download pre-trained face recognition models.

Downloads BlazeFace and MobileFaceNet models for the system.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
from loguru import logger

# Model URLs
MODELS = {
    'insightface': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        'filename': 'buffalo_l.zip',
        'extract_to': 'models/insightface'
    }
}


def download_file(url: str, destination: str):
    """Download file with progress."""
    logger.info(f"Downloading {url}")
    logger.info(f"Destination: {destination}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print()  # New line after progress
    logger.success(f"Downloaded to {destination}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    logger.info(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.success(f"Extracted to {extract_to}")


def download_insightface():
    """Download InsightFace models."""
    logger.info("Downloading InsightFace models...")

    # Create directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    info = MODELS['insightface']
    zip_path = models_dir / info['filename']
    extract_path = Path(info['extract_to'])

    # Download
    if not zip_path.exists():
        download_file(info['url'], str(zip_path))
    else:
        logger.info(f"Already downloaded: {zip_path}")

    # Extract
    if not extract_path.exists():
        extract_path.mkdir(parents=True, exist_ok=True)
        extract_zip(str(zip_path), str(extract_path))
    else:
        logger.info(f"Already extracted: {extract_path}")

    # List downloaded models
    logger.success("InsightFace models downloaded!")
    logger.info("Available models:")
    for model_file in extract_path.rglob("*.onnx"):
        logger.info(f"  - {model_file}")

    return True


def main():
    logger.info("=" * 60)
    logger.info("Model Download Script")
    logger.info("=" * 60)

    # Download InsightFace (includes face recognition model)
    download_insightface()

    logger.info("=" * 60)
    logger.success("All models downloaded successfully!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Models are in the 'models/' directory")
    logger.info("2. The system will automatically use these models")
    logger.info("3. Run: python scripts/enroll_face.py 'YourName'")

    return 0


if __name__ == "__main__":
    exit(main())
