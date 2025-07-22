import pytest
import os
import cv2
from PIL import Image
from pathlib import Path
import tempfile

from app import get_frames, process_video, process_user_input, process_history, extract_pdf_text

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
    
def test_process_history_basic_functionality():
    """Test basic conversation processing and content buffering."""
    # Empty history
    assert process_history([]) == []
    
    # Simple conversation
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    result = process_history(history)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    assert result[1] == {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
    assert result[2] == {"role": "user", "content": [{"type": "text", "text": "How are you?"}]}


def test_process_history_file_handling():
    """Test processing of different file types and content buffering."""
    # Create temp image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img:
        image_path = img.name
    
    video_path = os.path.join(ROOT_DIR, "assets", "test_video.mp4")
    
    try:
        history = [
            {"role": "user", "content": (image_path,)},
            {"role": "user", "content": "What's this image?"},
            {"role": "user", "content": (video_path,)},
            {"role": "assistant", "content": "I see an image and video."},
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"}  # Multiple user messages at end
        ]
        
        result = process_history(history)
        assert len(result) == 3
        
        # First user turn: image + text + video
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0] == {"type": "image", "url": image_path}
        assert result[0]["content"][1] == {"type": "text", "text": "What's this image?"}
        assert result[0]["content"][2] == {"type": "text", "text": "[Video uploaded previously]"}
        
        # Assistant response
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == [{"type": "text", "text": "I see an image and video."}]
        
        # Final user turn: multiple buffered messages
        assert result[2]["role"] == "user"
        assert len(result[2]["content"]) == 3
        assert result[2]["content"][0] == {"type": "text", "text": "First"}
        assert result[2]["content"][1] == {"type": "text", "text": "Second"}
        assert result[2]["content"][2] == {"type": "text", "text": "Third"}
        
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)


def test_extract_pdf_text_nonexistent_file():
    """Test that extract_pdf_text handles non-existent files appropriately."""
    with pytest.raises(ValueError, match="File not found"):
        extract_pdf_text("nonexistent_file.pdf")


def test_extract_pdf_text_with_mock_pdf():
    """Test PDF text extraction with a simple PDF file."""
    import fitz  # PyMuPDF
    
    # Create a temporary PDF with some text content
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        # Create a simple PDF with text
        doc = fitz.open()  # Create new PDF
        page = doc.new_page()
        
        # Add some text to the page
        text_content = "This is a test PDF document.\nIt contains multiple lines of text.\nPage 1 content here."
        page.insert_text((50, 100), text_content, fontsize=12)
        
        # Save the PDF
        doc.save(pdf_path)
        doc.close()
        
        # Test the extract_pdf_text function
        result = extract_pdf_text(pdf_path)
        
        # Verify the extracted text contains our content
        assert isinstance(result, str)
        assert "This is a test PDF document" in result
        assert "Page 1:" in result  # Should include page number
        assert "multiple lines of text" in result
        
    finally:
        # Clean up the temporary PDF file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_extract_pdf_text_empty_pdf():
    """Test PDF text extraction with an empty PDF (no text content)."""
    import fitz
    
    # Create a temporary empty PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        # Create an empty PDF
        doc = fitz.open()  # Create new PDF
        page = doc.new_page()  # Add empty page
        doc.save(pdf_path)
        doc.close()
        
        # Test the extract_pdf_text function
        result = extract_pdf_text(pdf_path)
        
        # Should return message about no content
        assert result == "No text content found in the PDF."
        
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_extract_pdf_text_multipage():
    """Test PDF text extraction with multiple pages."""
    import fitz
    
    # Create a temporary multi-page PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        # Create a PDF with multiple pages
        doc = fitz.open()
        
        # Page 1
        page1 = doc.new_page()
        page1.insert_text((50, 100), "Content from page one.", fontsize=12)
        
        # Page 2
        page2 = doc.new_page()
        page2.insert_text((50, 100), "Content from page two.", fontsize=12)
        
        # Page 3 (empty)
        page3 = doc.new_page()
        
        # Page 4
        page4 = doc.new_page()
        page4.insert_text((50, 100), "Content from page four.", fontsize=12)
        
        doc.save(pdf_path)
        doc.close()
        
        # Test the extract_pdf_text function
        result = extract_pdf_text(pdf_path)
        
        # Verify all pages with content are included
        assert "Page 1:" in result
        assert "Content from page one" in result
        assert "Page 2:" in result
        assert "Content from page two" in result
        assert "Page 4:" in result
        assert "Content from page four" in result
        
        # Page 3 should be excluded (empty)
        assert "Page 3:" not in result
        
        # Check that pages are separated properly
        assert "\n\n" in result  # Pages should be separated by double newlines
        
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_process_user_input_with_pdf():
    """Test processing user input with a PDF file."""
    import fitz
    
    # Create a temporary PDF for testing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        # Create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Test PDF content for user input processing.", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        # Test processing user input with PDF
        message = {
            "text": "Analyze this PDF",
            "files": [pdf_path]
        }
        
        result = process_user_input(message, 3)
        
        # Should have 2 items (original text + PDF content)
        assert len(result) == 2
        
        # First item should be the message text
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Analyze this PDF"
        
        # Second item should be PDF content
        assert result[1]["type"] == "text"
        assert "PDF Content:" in result[1]["text"]
        assert "Test PDF content for user input processing" in result[1]["text"]
        assert "Page 1:" in result[1]["text"]
        
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_process_user_input_pdf_error_handling():
    """Test that PDF processing errors are handled gracefully."""
    # Create a file that looks like a PDF but isn't valid
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(b"This is not a valid PDF file content")
        invalid_pdf_path = temp_file.name
    
    try:
        message = {
            "text": "Process invalid PDF",
            "files": [invalid_pdf_path]
        }
        
        result = process_user_input(message, 3)
        
        # Should have 2 items (original text + error message)
        assert len(result) == 2
        
        # First item should be the message text
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Process invalid PDF"
        
        # Second item should be error message
        assert result[1]["type"] == "text"
        assert "Error processing PDF:" in result[1]["text"]
        
    finally:
        # Clean up
        if os.path.exists(invalid_pdf_path):
            os.unlink(invalid_pdf_path)


def test_process_history_with_pdf():
    """Test that PDF files in history are handled correctly."""
    import fitz
    
    # Create a temporary PDF for testing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
    
    try:
        # Create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Historical PDF content.", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        # Test history with PDF file
        history = [
            {"role": "user", "content": (pdf_path,)},
            {"role": "user", "content": "What does this PDF contain?"},
            {"role": "assistant", "content": "The PDF contains some text."},
            {"role": "user", "content": "Thanks!"}
        ]
        
        result = process_history(history)
        
        # Should have 3 messages (user turn, assistant turn, final user turn)
        assert len(result) == 3
        
        # First user turn should have PDF placeholder and text
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "[PDF uploaded previously]"}
        assert result[0]["content"][1] == {"type": "text", "text": "What does this PDF contain?"}
        
        # Assistant response
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == [{"type": "text", "text": "The PDF contains some text."}]
        
        # Final user message
        assert result[2]["role"] == "user"
        assert result[2]["content"] == [{"type": "text", "text": "Thanks!"}]
        
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)