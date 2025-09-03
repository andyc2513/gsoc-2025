import os
import cv2
import fitz
import tempfile
from PIL import Image
from loguru import logger

# Constants
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB

PRESET_PROMPTS = {
    "General Assistant": "You are a helpful AI assistant capable of analyzing images, videos, and PDF documents. Provide clear, accurate, and helpful responses to user queries.",
    
    "Document Analyzer": "You are a specialized document analysis assistant. Focus on extracting key information, summarizing content, and answering specific questions about uploaded documents. For PDFs, provide structured analysis including main topics, key points, and relevant details. For images containing text, perform OCR-like analysis.",
    
    "Visual Content Expert": "You are an expert in visual content analysis. When analyzing images, provide detailed descriptions of visual elements, composition, colors, objects, people, and scenes. For videos, describe the sequence of events, movements, and changes between frames. Identify artistic techniques, styles, and visual storytelling elements.",
    
    "Educational Tutor": "You are a patient and encouraging educational tutor. Break down complex concepts into simple, understandable explanations. When analyzing educational materials (images, videos, or documents), focus on learning objectives, key concepts, and provide additional context or examples to enhance understanding.",
    
    "Technical Reviewer": "You are a technical expert specializing in analyzing technical documents, diagrams, code screenshots, and instructional videos. Provide detailed technical insights, identify potential issues, suggest improvements, and explain technical concepts with precision and accuracy.",
    
    "Creative Storyteller": "You are a creative storyteller who brings visual content to life through engaging narratives. When analyzing images or videos, create compelling stories, describe scenes with rich detail, and help users explore the creative and emotional aspects of visual content.",
}

def check_file_size(file_path: str) -> bool:
    """Check if a file meets the size requirements for processing.

    Validates that the file exists and is within the allowed size limits based on file type.
    Video files (.mp4, .mov) have a limit of 100MB, while image files have a limit of 10MB.

    Args:
        file_path (str): The absolute path to the file to be checked.

    Returns:
        bool: True if the file meets size requirements.

    Raises:
        ValueError: If the file doesn't exist, or if the file size exceeds the maximum
            allowed size for its type.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    
    if file_path.lower().endswith((".mp4", ".mov")):
        if file_size > MAX_VIDEO_SIZE:
            raise ValueError(f"Video file too large: {file_size / (1024*1024):.1f}MB. Maximum allowed: {MAX_VIDEO_SIZE / (1024*1024):.0f}MB")
    else:
        if file_size > MAX_IMAGE_SIZE:
            raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB. Maximum allowed: {MAX_IMAGE_SIZE / (1024*1024):.0f}MB")
    
    return True


def get_frames(video_path: str, max_images: int) -> list[tuple[Image.Image, float]]:
    """Extract frames from a video file at regular intervals.

    Opens a video file and extracts frames at evenly distributed intervals to get
    a representative sample of the video content. Each frame is converted to RGB
    format and returned as a PIL Image along with its timestamp.

    Args:
        video_path (str): The absolute path to the video file (.mp4 or .mov).
        max_images (int): The maximum number of frames to extract from the video.
            Must be a positive integer.

    Returns:
        list[tuple[Image.Image, float]]: A list of tuples where each tuple contains
            an Image.Image object (the extracted frame in RGB format) and a float
            (the timestamp of the frame in seconds, rounded to 2 decimal places).

    Raises:
        ValueError: If the video file cannot be opened or if file size validation fails.
    """
    check_file_size(video_path)
    
    frames: list[tuple[Image.Image, float]] = []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(total_frames // max_images, 1)
    max_position = min(total_frames, max_images * frame_interval)
    i = 0

    while i < max_position and len(frames) < max_images:
        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = capture.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))

        i += frame_interval

    capture.release()
    return frames


def process_video(video_path: str, max_images: int) -> list[dict]:
    """Process a video file and return formatted content for model input.

    Extracts frames from a video file, saves them as temporary PNG files, and
    formats them into a structure suitable for multimodal model input. Each frame
    is paired with descriptive text indicating its timestamp.

    Args:
        video_path (str): The absolute path to the video file to be processed.
        max_images (int): The maximum number of frames to extract and process.

    Returns:
        list[dict]: A list of dictionaries representing the processed video content.
            The structure alternates between text descriptions and image references:
            {"type": "text", "text": "Frame {timestamp}:"} and
            {"type": "image", "url": "/path/to/temp/frame.png"}.

    Note:
        Creates temporary PNG files that are not automatically cleaned up.
        The caller is responsible for cleanup if needed.
    """
    result_content = []
    frames = get_frames(video_path, max_images)
    for frame in frames:
        image, timestamp = frame
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            result_content.append({"type": "text", "text": f"Frame {timestamp}:"})
            result_content.append({"type": "image", "url": temp_file.name})
    logger.debug(
        f"Processed {len(frames)} frames from video {video_path} with frames {result_content}"
    )
    return result_content


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Opens a PDF file and extracts all readable text content from each page.
    Pages are numbered and formatted for readability. Empty pages are skipped.

    Args:
        pdf_path (str): The absolute path to the PDF file to be processed.

    Returns:
        str: The extracted text content with page numbers and formatting.
            If no text is found, returns a message indicating no content was found.

    Raises:
        ValueError: If the file size validation fails or if PDF processing encounters
            an error that prevents text extraction.
    """
    check_file_size(pdf_path)
    
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                text_content.append(f"Page {page_num + 1}:\n{text}")
        
        doc.close()
        
        if not text_content:
            return "No text content found in the PDF."
        
        return "\n\n".join(text_content)
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def process_user_input(message: dict, max_images: int) -> list[dict]:
    """Process user input including files and return formatted content for the model.

    Takes a user message that may contain text and file attachments, processes each
    file according to its type, and returns a structured format suitable for
    multimodal model input. Handles videos, PDFs, and image files.

    Args:
        message (dict): A dictionary containing user input with keys:
            "text" (str) - The user's text message, and
            "files" (list[str]) - List of file paths attached to the message.
        max_images (int): Maximum number of frames to extract from video files.

    Returns:
        list[dict]: A list of dictionaries representing the processed content with
            types "text" or "image" and corresponding content data. Includes error
            messages for files that cannot be processed.
    """
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]

    result_content = [{"type": "text", "text": message["text"]}]

    for file_path in message["files"]:
        try:
            check_file_size(file_path)
        except ValueError as e:
            logger.error(f"File size check failed: {e}")
            result_content.append({"type": "text", "text": f"Error: {str(e)}"})
            continue
            
        if file_path.endswith((".mp4", ".mov")):
            try:
                result_content = [*result_content, *process_video(file_path, max_images)]
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
                result_content.append({"type": "text", "text": f"Error processing video: {str(e)}"})
        elif file_path.lower().endswith(".pdf"):
            try:
                logger.info(f"Processing PDF file: {file_path}")
                pdf_text = extract_pdf_text(file_path)
                logger.debug(f"PDF text extracted successfully, length: {len(pdf_text)} characters")
                result_content.append({"type": "text", "text": f"PDF Content:\n{pdf_text}"})
            except ValueError as ve:
                logger.error(f"PDF validation failed: {ve}")
                result_content.append({"type": "text", "text": f"Error processing PDF: {str(ve)}"})
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                result_content.append({"type": "text", "text": f"Error processing PDF: {str(e)}"})
        else:
            result_content = [*result_content, {"type": "image", "url": file_path}]

    return result_content


def process_history(history: list[dict]) -> list[dict]:
    """Process chat history into the format expected by the model.

    Converts chat history from the UI format into the structured format required
    by multimodal language models. Groups consecutive user messages and handles
    different content types (text, images, videos, PDFs) appropriately.

    Args:
        history (list[dict]): A list of chat history items, where each item contains
            "role" (str) - either "user" or "assistant", and
            "content" - the message content (str for text, tuple for files).

    Returns:
        list[dict]: A list of messages formatted for the model with "role" and
            "content" keys, where content is a list of dictionaries with "type"
            and associated data.

    Note:
        Groups consecutive user messages into a single message. Videos and PDFs
        in history are replaced with placeholder text to avoid reprocessing.
    """
    messages = []
    content_buffer = []

    for item in history:
        if item["role"] == "assistant":
            if content_buffer:
                messages.append({"role": "user", "content": content_buffer})
                content_buffer = []

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["content"]}],
                }
            )
        else:
            content = item["content"]
            if isinstance(content, str):
                content_buffer.append({"type": "text", "text": content})
            elif isinstance(content, tuple) and len(content) > 0:
                file_path = content[0]
                if file_path.endswith((".mp4", ".mov")):
                    content_buffer.append({"type": "text", "text": "[Video uploaded previously]"})
                elif file_path.lower().endswith(".pdf"):
                    content_buffer.append({"type": "text", "text": "[PDF uploaded previously]"})
                else:
                    content_buffer.append({"type": "image", "url": file_path})

    if content_buffer:
        messages.append({"role": "user", "content": content_buffer})

    return messages


def update_custom_prompt(preset_choice: str) -> str:
    """Update the custom prompt based on preset selection.

    Returns the appropriate preset prompt text based on the user's selection.
    If "Custom Prompt" is selected, returns an empty string to allow manual input.

    Args:
        preset_choice (str): The name of the selected preset prompt. Should match
            one of the keys in PRESET_PROMPTS or be "Custom Prompt".

    Returns:
        str: The preset prompt text corresponding to the selection, or an empty
            string if "Custom Prompt" is selected or if the preset is not found.
    """
    if preset_choice == "Custom Prompt":
        return ""
    return PRESET_PROMPTS.get(preset_choice, "")


def get_preset_prompts() -> dict[str, str]:
    """Return the dictionary of preset prompts for the main application.

    Provides a copy of the predefined prompt templates that can be used throughout
    the application. Each preset is designed for a specific use case and contains
    detailed instructions for the AI model's behavior.

    Returns:
        dict[str, str]: A dictionary mapping preset names to their prompt texts.
            Includes prompts for general assistance, document analysis, visual content
            analysis, educational tutoring, technical review, and creative storytelling.

    Note:
        Returns a copy of the PRESET_PROMPTS dictionary to prevent accidental
        modification of the original constants.
    """
    return PRESET_PROMPTS.copy()