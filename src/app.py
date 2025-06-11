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
