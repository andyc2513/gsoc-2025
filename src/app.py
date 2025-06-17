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
    # TODO: Change max_image to slider
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

    if message["files"][0].endswith(".mp4"):
        return [
            {"type": "text", "text": message["text"]},
            *process_video(message["files"][0], max_images),
        ]

    return [
        {"type": "text", "text": message["text"]},
        *[{"type": "image", "url": path} for path in message["files"]],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    user_content_buffer = []

    for item in history:
        if item["role"] == "assistant":
            if user_content_buffer:
                messages.append({"role": "user", "content": user_content_buffer})
                user_content_buffer = []

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["content"]}],
                }
            )
        else:
            content = item["content"]
            user_content_buffer.append(
                {"type": "text", "text": content}
                if isinstance(content, str)
                else {"type": "image", "url": content[0]}
            )

    if user_content_buffer:
        messages.append({"role": "user", "content": user_content_buffer})

    return messages
