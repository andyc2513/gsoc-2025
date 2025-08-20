import torch
torch._dynamo.config.disable = True
from collections.abc import Iterator
from transformers import (
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
    Gemma3Processor,
    Gemma3nForConditionalGeneration,
)
import spaces
import tempfile
from threading import Thread
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv
import cv2
from loguru import logger
from PIL import Image
import fitz

dotenv_path = find_dotenv()

load_dotenv(dotenv_path)

model_12_id = os.getenv("MODEL_12_ID", "google/gemma-3-12b-it")
model_3n_id = os.getenv("MODEL_3N_ID", "google/gemma-3n-E4B-it")

MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB

input_processor = Gemma3Processor.from_pretrained(model_12_id)

model_12 = Gemma3ForConditionalGeneration.from_pretrained(
    model_12_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

model_3n = Gemma3nForConditionalGeneration.from_pretrained(
    model_3n_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)


def check_file_size(file_path: str) -> bool:
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
        return ValueError(f"Failed to extract text from PDF: {str(e)}")


def process_user_input(message: dict, max_images: int) -> list[dict]:
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
                pdf_text = extract_pdf_text(file_path)
                result_content.append({"type": "text", "text": f"PDF Content:\n{pdf_text}"})
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                result_content.append({"type": "text", "text": f"Error processing PDF: {str(e)}"})
        else:
            result_content = [*result_content, {"type": "image", "url": file_path}]

    return result_content

def process_history(history: list[dict]) -> list[dict]:
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


@spaces.GPU(duration=120)
def run(
    message: dict,
    history: list[dict],
    system_prompt_preset: str,
    custom_system_prompt: str,
    model_choice: str,
    max_new_tokens: int,
    max_images: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[str]:

    # Define preset system prompts
    preset_prompts = {
        "General Assistant": "You are a helpful AI assistant capable of analyzing images, videos, and PDF documents. Provide clear, accurate, and helpful responses to user queries.",
        
        "Document Analyzer": "You are a specialized document analysis assistant. Focus on extracting key information, summarizing content, and answering specific questions about uploaded documents. For PDFs, provide structured analysis including main topics, key points, and relevant details. For images containing text, perform OCR-like analysis.",
        
        "Visual Content Expert": "You are an expert in visual content analysis. When analyzing images, provide detailed descriptions of visual elements, composition, colors, objects, people, and scenes. For videos, describe the sequence of events, movements, and changes between frames. Identify artistic techniques, styles, and visual storytelling elements.",
        
        "Educational Tutor": "You are a patient and encouraging educational tutor. Break down complex concepts into simple, understandable explanations. When analyzing educational materials (images, videos, or documents), focus on learning objectives, key concepts, and provide additional context or examples to enhance understanding.",
        
        "Technical Reviewer": "You are a technical expert specializing in analyzing technical documents, diagrams, code screenshots, and instructional videos. Provide detailed technical insights, identify potential issues, suggest improvements, and explain technical concepts with precision and accuracy.",
        
        "Creative Storyteller": "You are a creative storyteller who brings visual content to life through engaging narratives. When analyzing images or videos, create compelling stories, describe scenes with rich detail, and help users explore the creative and emotional aspects of visual content.",
    }
    
    # Determine which system prompt to use
    if system_prompt_preset == "Custom Prompt":
        system_prompt = custom_system_prompt
    else:
        system_prompt = preset_prompts.get(system_prompt_preset, custom_system_prompt)

    logger.debug(
        f"\n message: {message} \n history: {history} \n system_prompt_preset: {system_prompt_preset} \n "
        f"system_prompt: {system_prompt} \n model_choice: {model_choice} \n max_new_tokens: {max_new_tokens} \n max_images: {max_images}"
    )

    def try_fallback_model(original_model_choice: str):
        fallback_model = model_3n if original_model_choice == "Gemma 3 12B" else model_12
        fallback_name = "Gemma 3n E4B" if original_model_choice == "Gemma 3 12B" else "Gemma 3 12B"
        logger.info(f"Attempting fallback to {fallback_name} model")
        return fallback_model, fallback_name

    selected_model = model_12 if model_choice == "Gemma 3 12B" else model_3n
    current_model_name = model_choice

    try:
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.extend(process_history(history))
        messages.append(
            {"role": "user", "content": process_user_input(message, max_images)}
        )

        inputs = input_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=selected_model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(
            input_processor, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )
        
        t = Thread(target=selected_model.generate, kwargs=generate_kwargs)
        t.start()

        output = ""
        generation_failed = False
        
        try:
            for delta in streamer:
                if delta is None:
                    continue
                output += delta
                yield output
                
        except Exception as stream_error:
            logger.error(f"Streaming failed with {current_model_name}: {stream_error}")
            generation_failed = True
            
        # Wait for thread to complete
        t.join(timeout=120)  # 2 minute timeout
        
        if t.is_alive() or generation_failed or not output.strip():
            raise Exception(f"Generation failed or timed out with {current_model_name}")
            
    except Exception as primary_error:
        logger.error(f"Primary model ({current_model_name}) failed: {primary_error}")
        
        # Try fallback model
        try:
            selected_model, fallback_name = try_fallback_model(model_choice)
            logger.info(f"Switching to fallback model: {fallback_name}")
            
            # Rebuild inputs for fallback model
            inputs = input_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device=selected_model.device, dtype=torch.bfloat16)

            streamer = TextIteratorStreamer(
                input_processor, skip_prompt=True, skip_special_tokens=True, timeout=60.0
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
            
            t = Thread(target=selected_model.generate, kwargs=generate_kwargs)
            t.start()

            output = f"⚠️ Switched to {fallback_name} due to {current_model_name} failure.\n\n"
            yield output
            
            try:
                for delta in streamer:
                    if delta is None:
                        continue
                    output += delta
                    yield output
            except Exception as fallback_stream_error:
                logger.error(f"Fallback streaming failed: {fallback_stream_error}")
                raise fallback_stream_error
                
            # Wait for fallback thread
            t.join(timeout=120)
            
            if t.is_alive() or not output.strip():
                raise Exception(f"Fallback model {fallback_name} also failed")
                
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            
            # Final fallback - return error message
            error_message = (
                "❌ **Generation Failed**\n\n"
                f"Both {model_choice} and fallback model encountered errors. "
                "This could be due to:\n"
                "- High server load\n"
                "- Memory constraints\n"
                "- Input complexity\n\n"
                "**Suggestions:**\n"
                "- Try reducing max tokens or image count\n"
                "- Simplify your prompt\n"
                "- Try again in a few moments\n\n"
                f"*Error details: {str(primary_error)[:200]}...*"
            )
            yield error_message


demo = gr.ChatInterface(
    fn=run,
    type="messages",
    chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
    textbox=gr.MultimodalTextbox(
        file_types=[".mp4", ".jpg", ".png", ".pdf"], file_count="multiple", autofocus=True
    ),
    multimodal=True,
    additional_inputs=[
        gr.Dropdown(
            label="System Prompt Preset",
            choices=[
                "General Assistant",
                "Document Analyzer", 
                "Visual Content Expert",
                "Educational Tutor",
                "Technical Reviewer",
                "Creative Storyteller",
                "Custom Prompt"
            ],
            value="General Assistant",
            info="System prompts define the AI's role and behavior. Choose a preset that matches your task, or select 'Custom Prompt' to write your own specialized instructions."
        ),
        gr.Textbox(
            label="Custom System Prompt", 
            value="You are a helpful AI assistant capable of analyzing images, videos, and PDF documents. Provide clear, accurate, and helpful responses to user queries.",
            lines=3,
            info="Edit this field when 'Custom Prompt' is selected above, or modify any preset"
        ),
        gr.Dropdown(
            label="Model",
            choices=["Gemma 3 12B", "Gemma 3n E4B"],
            value="Gemma 3 12B",
            info="Gemma 3 12B: More powerful and detailed responses, but slower processing. Gemma 3n E4B: Faster processing with efficient performance for most tasks."
        ),
        gr.Slider(
            label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700
        ),
        gr.Slider(label="Max Images", minimum=1, maximum=4, step=1, value=2),
        gr.Slider(
            label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.7
        ),
        gr.Slider(
            label="Top P", minimum=0.1, maximum=1.0, step=0.05, value=0.9
        ),
        gr.Slider(
            label="Top K", minimum=1, maximum=100, step=1, value=50
        ),
        gr.Slider(
            label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1
        )
    ],
    stop_btn=False,
)

# Add JavaScript to update custom prompt when preset changes
def update_custom_prompt(preset_choice):
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

# Connect the dropdown to update the textbox
with demo:
    preset_dropdown = demo.additional_inputs[0]
    custom_textbox = demo.additional_inputs[1]
    preset_dropdown.change(
        fn=update_custom_prompt,
        inputs=[preset_dropdown],
        outputs=[custom_textbox]
    )

if __name__ == "__main__":
    demo.launch()
