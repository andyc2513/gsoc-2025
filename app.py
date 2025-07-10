import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import (
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
    Gemma3Processor,
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

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MODEL_CONFIGS = {
    "Gemma 3 4B IT": {
        "id": os.getenv("MODEL_ID_27", "google/gemma-3-4b-it"),
        "supports_video": True,
        "supports_pdf": False
    },
    "Gemma 3 1B IT": {
        "id": os.getenv("MODEL_ID_12", "google/gemma-3-1b-it"), 
        "supports_video": True,
        "supports_pdf": False
    },
    "Gemma 3N E4B IT": {
        "id": os.getenv("MODEL_ID_3N", "google/gemma-3n-E4B-it"),
        "supports_video": False,
        "supports_pdf": False
    }
}

# Load all models and processors
models = {}
processor = Gemma3Processor.from_pretrained("google/gemma-3-4b-it")

for model_name, config in MODEL_CONFIGS.items():
    logger.info(f"Loading {model_name}...")
    
    models[model_name] = Gemma3ForConditionalGeneration.from_pretrained(
        config["id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    
    logger.info(f"✓ {model_name} loaded successfully")

# Current model selection (default)
current_model = "Gemma 3 27B IT"

def get_frames(video_path: str, max_images: int) -> list[tuple[Image.Image, float]]:
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


def process_user_input(message: dict, max_images: int) -> list[dict]:
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]

    result_content = [{"type": "text", "text": message["text"]}]

    for file_path in message["files"]:
        if file_path.endswith((".mp4", ".mov")):
            result_content = [*result_content, *process_video(file_path, max_images)]
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
                else:
                    content_buffer.append({"type": "image", "url": file_path})

    if content_buffer:
        messages.append({"role": "user", "content": content_buffer})

    return messages


def get_supported_file_types(model_name: str) -> list[str]:
    """Get supported file types for the selected model."""
    config = MODEL_CONFIGS[model_name]
    
    base_types = [".jpg", ".png", ".jpeg", ".gif", ".bmp", ".webp"]
    
    if config["supports_video"]:
        base_types.extend([".mp4", ".mov", ".avi"])
    
    if config["supports_pdf"]:
        base_types.append(".pdf")
    
    return base_types

@spaces.GPU(duration=120)
def run(
    message: dict,
    history: list[dict],
    model_name: str,
    system_prompt: str,
    max_new_tokens: int,
    max_images: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[str]:
    
    global current_model
    
    if model_name != current_model:
        current_model = model_name
        logger.info(f"Switched to model: {model_name}")
    
    logger.debug(
        f"\n message: {message} \n history: {history} \n model: {model_name} \n "
        f"system_prompt: {system_prompt} \n max_new_tokens: {max_new_tokens} \n max_images: {max_images}"
    )

    config = MODEL_CONFIGS[model_name]
    if not config["supports_video"] and message.get("files"):
        for file_path in message["files"]:
            if file_path.endswith((".mp4", ".mov", ".avi")):
                yield "Error: Selected model does not support video files. Please choose a video-capable model."
                return

    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.extend(process_history(history))
    messages.append(
        {"role": "user", "content": process_user_input(message, max_images)}
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device=models[current_model].device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(
        processor, timeout=60.0, skip_prompt=True, skip_special_tokens=True
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
    t = Thread(target=models[current_model].generate, kwargs=generate_kwargs)
    t.start()

    output = ""
    for delta in streamer:
        output += delta
        yield output

def create_interface():
    """Create interface with model selector."""
    
    initial_file_types = get_supported_file_types(current_model)
    
    demo = gr.ChatInterface(
        fn=run,
        type="messages",
        chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
        textbox=gr.MultimodalTextbox(
            file_types=initial_file_types, 
            file_count="multiple", 
            autofocus=True
        ),
        multimodal=True,
        additional_inputs=[
            gr.Dropdown(
                label="Model",
                choices=list(MODEL_CONFIGS.keys()),
                value=current_model,
                info="Select which model to use for generation"
            ),
            gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
            gr.Slider(
                label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700
            ),
            gr.Slider(label="Max Images", minimum=1, maximum=8, step=1, value=2),
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
            ),
        ],
        stop_btn=False,
        title="Multi-Model Gemma Chat"
    )
    
    return demo

demo = create_interface()

if __name__ == "__main__":
    demo.launch()
