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

model_id = os.getenv("MODEL_ID", "google/gemma-3-4b-it")

input_processor = Gemma3Processor.from_pretrained(model_id)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)


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
            content_buffer.append(
                {"type": "text", "text": content}
                if isinstance(content, str)
                else {"type": "image", "url": content[0]}
            )

    if content_buffer:
        messages.append({"role": "user", "content": content_buffer})

    return messages


@spaces.GPU(duration=120)
def run(
    message: dict,
    history: list[dict],
    system_prompt: str,
    max_new_tokens: int,
    max_images: int,
) -> Iterator[str]:

    logger.debug(
        f"\n message: {message} \n history: {history} \n system_prompt: {system_prompt} \n "
        f"max_new_tokens: {max_new_tokens} \n max_images: {max_images}"
    )

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
    ).to(device=model.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(
        input_processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
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
        file_types=[".mp4", ".img"], file_count="multiple", autofocus=True
    ),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
        gr.Slider(
            label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700
        ),
        gr.Slider(label="Max Images", minimum=1, maximum=4, step=1, value=2),
    ],
    stop_btn=False,
)

if __name__ == "__main__":
    demo.launch()
