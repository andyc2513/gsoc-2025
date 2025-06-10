import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import Gemma3ForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import spaces
from threading import Thread
import gradio as gr
import os
import cv2
import gradio as gr
import spaces
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer