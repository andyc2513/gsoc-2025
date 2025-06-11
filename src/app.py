import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
import spaces
from threading import Thread
import gradio as gr
import os
import cv2
from loguru import logger
from PIL import Image