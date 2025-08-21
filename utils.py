import os
import cv2
import fitz
import tempfile
from PIL import Image
from loguru import logger

# Constants
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB


def check_file_size(file_path: str) -> bool:
    """Check if a file meets the size requirements."""
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
    """Extract frames from a video file."""
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
    """Process a video file and return formatted content for the model."""
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
    """Extract text content from a PDF file."""
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
    """Process user input including files and return formatted content for the model."""
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
    """Process chat history into the format expected by the model."""
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


def update_custom_prompt(preset_choice):
    """Update the custom prompt based on preset selection."""
    preset_prompts = {
        "General Assistant": "You are a helpful AI assistant capable of analyzing images, videos, and PDF documents. Provide clear, accurate, and helpful responses to user queries.",
        
        "Document Analyzer": "You are a specialized document analysis assistant. Focus on extracting key information, summarizing content, and answering specific questions about uploaded documents. For PDFs, provide structured analysis including main topics, key points, and relevant details. For images containing text, perform OCR-like analysis.",
        
        "Visual Content Expert": "You are an expert in visual content analysis. When analyzing images, provide detailed descriptions of visual elements, composition, colors, objects, people, and scenes. For videos, describe the sequence of events, movements, and changes between frames. Identify artistic techniques, styles, and visual storytelling elements.",
        
        "Educational Tutor": "You are a patient and encouraging educational tutor. Break down complex concepts into simple, understandable explanations. When analyzing educational materials (images, videos, or documents), focus on learning objectives, key concepts, and provide additional context or examples to enhance understanding.",
        
        "Technical Reviewer": "You are a technical expert specializing in analyzing technical documents, diagrams, code screenshots, and instructional videos. Provide detailed technical insights, identify potential issues, suggest improvements, and explain technical concepts with precision and accuracy.",
        
        "Creative Storyteller": "You are a creative storyteller who brings visual content to life through engaging narratives. When analyzing images or videos, create compelling stories, describe scenes with rich detail, and help users explore the creative and emotional aspects of visual content.",
        
        "Custom Prompt": ""
    }
    
    return preset_prompts.get(preset_choice, "")


def get_preset_prompts():
    """Return the dictionary of preset prompts for the main application."""
    return {
        "General Assistant": "You are a helpful AI assistant capable of analyzing images, videos, and PDF documents. Provide clear, accurate, and helpful responses to user queries.",
        
        "Document Analyzer": "You are a specialized document analysis assistant. Focus on extracting key information, summarizing content, and answering specific questions about uploaded documents. For PDFs, provide structured analysis including main topics, key points, and relevant details. For images containing text, perform OCR-like analysis.",
        
        "Visual Content Expert": "You are an expert in visual content analysis. When analyzing images, provide detailed descriptions of visual elements, composition, colors, objects, people, and scenes. For videos, describe the sequence of events, movements, and changes between frames. Identify artistic techniques, styles, and visual storytelling elements.",
        
        "Educational Tutor": "You are a patient and encouraging educational tutor. Break down complex concepts into simple, understandable explanations. When analyzing educational materials (images, videos, or documents), focus on learning objectives, key concepts, and provide additional context or examples to enhance understanding.",
        
        "Technical Reviewer": "You are a technical expert specializing in analyzing technical documents, diagrams, code screenshots, and instructional videos. Provide detailed technical insights, identify potential issues, suggest improvements, and explain technical concepts with precision and accuracy.",
        
        "Creative Storyteller": "You are a creative storyteller who brings visual content to life through engaging narratives. When analyzing images or videos, create compelling stories, describe scenes with rich detail, and help users explore the creative and emotional aspects of visual content.",
    }
