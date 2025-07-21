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
    system_prompt: str,
    model_choice: str,
    max_new_tokens: int,
    max_images: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[str]:

    logger.debug(
        f"\n message: {message} \n history: {history} \n system_prompt: {system_prompt} \n "
        f"model_choice: {model_choice} \n max_new_tokens: {max_new_tokens} \n max_images: {max_images}"
    )

    selected_model = model_12 if model_choice == "Gemma 3 12B" else model_3n

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
    for delta in streamer:
        output += delta
        yield output


demo = gr.ChatInterface(
    fn=run,
    type="messages",
    chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
    textbox=gr.MultimodalTextbox(
        file_types=[".mp4", ".jpg", ".png", ".pdf"], file_count="multiple", autofocus=True
    ),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
        gr.Dropdown(
            label="Model",
            choices=["Gemma 3 12B", "Gemma 3n E4B"],
            value="Gemma 3 12B"
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

if __name__ == "__main__":
    demo.launch()
