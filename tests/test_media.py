import pytest
import os
import cv2
from PIL import Image
from pathlib import Path
import tempfile

from src.app import get_frames, process_video, process_user_input, process_history

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
        
def test_process_user_input_text_only():
    """Test processing user input with text only (no files)."""
    message = {
        "text": "This is a test message",
        "files": []
    }
    
    # Add the max_images parameter
    result = process_user_input(message, 5)
    
    # Should return a single text item
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "This is a test message"


def test_process_user_input_with_video():
    """Test processing user input with a video file."""
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    assert os.path.exists(video_path), f"Test video not found at {video_path}"
    
    message = {
        "text": "Video analysis",
        "files": [video_path]
    }
    
    result = process_user_input(message, 4)
    
    # Should have at least 3 items (text + at least one frame with text and image)
    assert len(result) >= 3
    
    # First item should be the message text
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Video analysis"
    
    # Following items should be frame text and images
    assert result[1]["type"] == "text"
    assert result[1]["text"].startswith("Frame ")
    
    assert result[2]["type"] == "image"
    assert "url" in result[2]
    assert os.path.exists(result[2]["url"])


def test_process_user_input_with_images():
    """Test processing user input with image files."""
    # Create temporary image files for testing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img2:
        
        image_paths = [img1.name, img2.name]
        
        message = {
            "text": "Image analysis",
            "files": image_paths
        }
        
        result = process_user_input(message, 5)
        
        # Should have 3 items (text + 2 images)
        assert len(result) == 3
        
        # First item should be the message text
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Image analysis"
        
        # Following items should be images
        assert result[1]["type"] == "image"
        assert result[1]["url"] == image_paths[0]
        
        assert result[2]["type"] == "image"
        assert result[2]["url"] == image_paths[1]
        
    # Clean up temp files
    for path in image_paths:
        if os.path.exists(path):
            os.unlink(path)


def test_process_user_input_empty_text():
    """Test processing user input with empty text but with files."""
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    
    message = {
        "text": "",  # Empty text
        "files": [video_path]
    }
    
    # Add max_images parameter
    result = process_user_input(message, 3)
    
    # First item should be empty text
    assert result[0]["type"] == "text"
    assert result[0]["text"] == ""
    
    # Rest should be video frames
    assert len(result) > 1


def test_process_user_input_handles_empty_files_list():
    """Test that an empty files list is handled correctly."""
    message = {
        "text": "No files",
        "files": []
    }
    
    # Add max_images parameter
    result = process_user_input(message, 3)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "No files"


def test_process_user_input_max_images_effect():
    """Test that max_images parameter correctly limits the number of frames."""
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    
    message = {
        "text": "Video with few frames",
        "files": [video_path]
    }
    
    result_few = process_user_input(message, 2)
    result_many = process_user_input(message, 5)
    
    # Count actual frames (each frame has a text and image entry)
    frames_few = (len(result_few) - 1) // 2  # -1 for initial text message
    frames_many = (len(result_many) - 1) // 2
    
    # Should respect max_images parameter
    assert frames_few <= 2
    assert frames_many <= 5
    assert frames_few < frames_many
    
def test_empty_history():
    """Test processing empty history."""
    history = []
    result = process_history(history)
    assert result == []

def test_single_user_message_text():
    """Test processing a single user text message."""
    history = [
        {"role": "user", "content": "Hello, AI!"}
    ]
    
    result = process_history(history)
    
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Hello, AI!"

def test_single_user_message_image():
    """Test processing a single user image message."""
    history = [
        {"role": "user", "content": ["path/to/image.jpg"]}
    ]
    
    result = process_history(history)
    
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "image"
    assert result[0]["content"][0]["url"] == "path/to/image.jpg"

def test_single_assistant_message():
    """Test processing a single assistant message."""
    history = [
        {"role": "assistant", "content": "I'm an AI assistant."}
    ]
    
    result = process_history(history)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "I'm an AI assistant."

def test_alternating_messages():
    """Test processing alternating user and assistant messages."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"}
    ]
    
    result = process_history(history)
    
    assert len(result) == 4
    assert [item["role"] for item in result] == ["user", "assistant", "user", "assistant"]
    assert result[0]["content"][0]["text"] == "Hello"
    assert result[1]["content"][0]["text"] == "Hi there"
    assert result[2]["content"][0]["text"] == "How are you?"
    assert result[3]["content"][0]["text"] == "I'm doing well, thanks!"

def test_consecutive_user_messages():
    """Test processing consecutive user messages - they should be grouped."""
    history = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {"role": "user", "content": "Third message"},
        {"role": "assistant", "content": "I got your messages"}
    ]
    
    result = process_history(history)
    
    # Should be combined into a single user message with multiple content items
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 3
    assert [item["text"] for item in result[0]["content"]] == [
        "First message", "Second message", "Third message"
    ]

def test_mixed_content_types():
    """Test processing mixed content types (text and images)."""
    history = [
        {"role": "user", "content": "Look at this:"},
        {"role": "user", "content": ["image.jpg"]},
        {"role": "assistant", "content": "Nice image!"}
    ]
    
    result = process_history(history)
    
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][1]["type"] == "image"

def test_ending_with_user_messages():
    """Test history that ends with user messages."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Another question"},
        {"role": "user", "content": ["image.png"]}
    ]
    
    result = process_history(history)
    
    assert len(result) == 3
    assert result[2]["role"] == "user"
    assert len(result[2]["content"]) == 2
    assert result[2]["content"][0]["type"] == "text"
    assert result[2]["content"][1]["type"] == "image"

def test_empty_messages():
    """Test handling of empty content messages."""
    history = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Hello"}
    ]
    
    result = process_history(history)
    
    assert len(result) == 3
    assert result[0]["content"][0]["text"] == ""
    assert result[1]["content"][0]["text"] == ""
    assert result[2]["content"][0]["text"] == "Hello"
