import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import (
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
    Gemma3Processor,
    Gemma3nForConditionalGeneration,
    Gemma3ForCausalLM
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

model_27_id = os.getenv("MODEL_27_ID", "google/gemma-3-4b-it")
model_12_id = os.getenv("MODEL_12_ID", "google/gemma-3-4b-it")
model_3n_id = os.getenv("MODEL_3N_ID", "google/gemma-3-4b-it")

input_processor = Gemma3Processor.from_pretrained(model_27_id)

model_27 = Gemma3ForConditionalGeneration.from_pretrained(
    model_27_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

model_12 = Gemma3ForCausalLM.from_pretrained(
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


@spaces.GPU(duration=120)
def run(
    message: dict,
    history: list[dict],
    system_prompt: str,
    max_new_tokens: int,
    max_images: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
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
    ).to(device=model_27.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(
        input_processor, timeout=60.0, skip_prompt=True, skip_special_tokens=True
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
    t = Thread(target=model_27.generate, kwargs=generate_kwargs)
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
        file_types=[".mp4", ".jpg", ".png"], file_count="multiple", autofocus=True
    ),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
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
