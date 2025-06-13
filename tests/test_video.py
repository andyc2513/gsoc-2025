import pytest
import os
import cv2
from PIL import Image
from pathlib import Path

from src.app import get_frames

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

def test_correct_frame_return():
    """Test that get_frames returns a list of (Image, float) tuples."""
    # Path to a test video file
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    
    # Ensure the test video exists
    assert os.path.exists(video_path), f"Test video not found at {video_path}"
    
    # Test with a small number of frames
    max_images = 3
    frames = get_frames(video_path, max_images)
    
    # Check return type
    assert isinstance(frames, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in frames)
    assert all(isinstance(img, Image.Image) and isinstance(ts, float) for img, ts in frames)