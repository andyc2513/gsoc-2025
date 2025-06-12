import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import Gemma3ForConditionalGeneration, TextIteratorStreamer, Gemma3Processor
import spaces
from threading import Thread
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv
import cv2
from loguru import logger
from PIL import Image

MAX_NUM_IMAGES = 6

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

def get_frames(video_path: str) -> list[tuple[Image.Image, float]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(total_frames // MAX_NUM_IMAGES, 1)
    frames: list[tuple[Image.Image, float]] = []

    for i in range(0, min(total_frames, MAX_NUM_IMAGES * frame_interval), frame_interval):
        if len(frames) >= MAX_NUM_IMAGES:
            break

        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = capture.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))

    capture.release()
    return frames