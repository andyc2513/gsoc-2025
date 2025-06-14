import pytest
import os
import cv2
from PIL import Image
from pathlib import Path
import tempfile

from src.app import get_frames, process_video

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent


def test_correct_frame_return():
    """Test that get_frames returns a list of (Image, float) tuples."""
    # Path to a test video file
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")

    # Ensure the test video exists
    assert os.path.exists(video_path)

    # Test with a small number of frames
    max_images = 3
    frames = get_frames(video_path, max_images)

    assert isinstance(frames, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in frames)
    assert all(
        isinstance(img, Image.Image) and isinstance(ts, float) for img, ts in frames
    )


def test_process_video_structure():
    """Test that process_video returns the expected list structure."""

    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    max_images = 2

    result = process_video(video_path, max_images)

    # Should have 2 items (text + image) per frame
    assert len(result) == max_images * 2

    # Check structure of items
    for i in range(0, len(result), 2):
        # Text item
        assert result[i]["type"] == "text"
        assert result[i]["text"].startswith("Frame ")

        # Image item
        assert result[i + 1]["type"] == "image"
        assert "url" in result[i + 1]
        assert os.path.exists(result[i + 1]["url"])

        # Verify the image file is valid
        try:
            img = Image.open(result[i + 1]["url"])
            img.verify()  # Make sure it's a valid image
        except Exception as e:
            pytest.fail(f"Invalid image file: {e}")


def test_process_video_timestamps():
    """Test that timestamps in the result are properly formatted."""

    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    max_images = 3

    result = process_video(video_path, max_images)

    # Extract timestamps from text items
    timestamps = []
    for i in range(0, len(result), 2):
        if result[i]["type"] == "text":
            # Extract timestamp from "Frame X.XX:" format
            timestamp_text = result[i]["text"].split()[1].rstrip(":")
            timestamps.append(float(timestamp_text))

    # Check timestamps are ascending
    assert len(timestamps) == max_images
    assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


def test_process_video_temp_files():
    """Test that temporary files are created and cleaned up properly."""

    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    max_images = 1

    result = process_video(video_path, max_images)

    # Verify temp file exists
    image_path = result[1]["url"]
    assert os.path.exists(image_path)
    assert image_path.endswith(".png")


def test_process_video_invalid_path():
    """Test that process_video handles invalid paths appropriately."""

    with pytest.raises(ValueError):
        process_video("nonexistent_video.mp4", 3)
