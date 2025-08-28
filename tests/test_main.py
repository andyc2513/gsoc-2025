import pytest
import os
import cv2
from PIL import Image
from pathlib import Path
import tempfile

from utils import get_frames, process_video, process_user_input, process_history, extract_pdf_text, update_custom_prompt, check_file_size, MAX_VIDEO_SIZE, MAX_IMAGE_SIZE

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent


def test_get_frames():
    """Test that get_frames returns correct structure and handles video processing."""
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    assert os.path.exists(video_path)

    frames = get_frames(video_path, 3)
    assert isinstance(frames, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in frames)
    assert all(isinstance(img, Image.Image) and isinstance(ts, float) for img, ts in frames)


def test_process_video():
    """Test video processing returns expected structure."""
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    result = process_video(video_path, 2)

    assert len(result) == 4  # 2 frames * 2 items per frame
    assert result[0]["type"] == "text" and result[0]["text"].startswith("Frame ")
    assert result[1]["type"] == "image" and os.path.exists(result[1]["url"])


def test_process_video_invalid_path():
    """Test that process_video handles invalid paths appropriately."""
    with pytest.raises(ValueError):
        process_video("nonexistent_video.mp4", 3)
        
def test_process_user_input():
    """Test processing user input with different file types."""
    # Text only
    message = {"text": "Test message", "files": []}
    result = process_user_input(message, 5)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    
    # With video
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    if os.path.exists(video_path):
        message = {"text": "Video analysis", "files": [video_path]}
        result = process_user_input(message, 2)
        assert len(result) >= 3  # text + frames
        assert result[0]["text"] == "Video analysis"
        assert result[1]["text"].startswith("Frame ")


def test_process_history():
    """Test basic conversation processing."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    result = process_history(history)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    assert result[1] == {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}


def test_extract_pdf_text():
    """Test PDF text extraction."""
    import fitz
    
    # Test non-existent file
    with pytest.raises(ValueError, match="File not found"):
        extract_pdf_text("nonexistent_file.pdf")
    
    # Test with valid PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Test PDF content.", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        result = extract_pdf_text(pdf_path)
        assert "Test PDF content" in result
        assert "Page 1:" in result
        
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_process_user_input_with_pdf():
    """Test processing user input with PDF."""
    import fitz
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Test PDF content.", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        message = {"text": "Analyze PDF", "files": [pdf_path]}
        result = process_user_input(message, 3)
        
        assert len(result) == 2
        assert result[0]["text"] == "Analyze PDF"
        assert "PDF Content:" in result[1]["text"]
        assert "Test PDF content" in result[1]["text"]
        
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_process_user_input_pdf_error_handling():
    """Test PDF error handling."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(b"Invalid PDF content")
        invalid_pdf_path = temp_file.name
    
    try:
        message = {"text": "Process invalid PDF", "files": [invalid_pdf_path]}
        result = process_user_input(message, 3)
        
        assert len(result) == 2
        assert result[0]["text"] == "Process invalid PDF"
        assert "Failed to extract text from PDF:" in result[1]["text"]
        
    finally:
        if os.path.exists(invalid_pdf_path):
            os.unlink(invalid_pdf_path)


def test_update_custom_prompt():
    """Test system prompt selection."""
    # Test key prompts
    general = update_custom_prompt("General Assistant")
    assert "images" in general.lower() and "videos" in general.lower()
    
    document = update_custom_prompt("Document Analyzer")
    assert "document" in document.lower() and "analysis" in document.lower()
    
    # Test custom returns empty
    assert update_custom_prompt("Custom") == ""
    assert update_custom_prompt("Invalid") == ""


def test_check_file_size():
    """Test file size validation."""
    # Test non-existent file
    with pytest.raises(ValueError, match="File not found"):
        check_file_size("nonexistent_file.txt")
    
    # Test valid small files
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(b"small content")
        temp_path = temp_file.name
    
    try:
        assert check_file_size(temp_path) is True
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test oversized image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_file.write(b"x" * (MAX_IMAGE_SIZE + 1024))
        temp_path = temp_file.name
    
    try:
        with pytest.raises(ValueError, match="Image file too large"):
            check_file_size(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)