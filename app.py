import torch
torch._dynamo.config.disable = True
from collections.abc import Iterator
from transformers import (
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
    Gemma3Processor,
    Gemma3nForConditionalGeneration,
    Gemma3nProcessor
)
import spaces
from threading import Thread
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from utils import *

dotenv_path = find_dotenv()

load_dotenv(dotenv_path)

model_12_id = os.getenv("MODEL_12_ID", "google/gemma-3-12b-it")
model_3n_id = os.getenv("MODEL_3N_ID", "google/gemma-3n-E4B-it")

input_processor_12 = Gemma3Processor.from_pretrained(model_12_id)
input_processor_3n = Gemma3nProcessor.from_pretrained(model_3n_id)

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
    preset_prompts = get_preset_prompts()
    
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
        fallback_processor = input_processor_3n if original_model_choice == "Gemma 3 12B" else input_processor_12
        fallback_name = "Gemma 3n E4B" if original_model_choice == "Gemma 3 12B" else "Gemma 3 12B"
        logger.info(f"Attempting fallback to {fallback_name} model")
        return fallback_model, fallback_processor, fallback_name

    selected_model = model_12 if model_choice == "Gemma 3 12B" else model_3n
    selected_processor = input_processor_12 if model_choice == "Gemma 3 12B" else input_processor_3n
    current_model_name = model_choice

    try:
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.extend(process_history(history))
        user_content = process_user_input(message, max_images)
        messages.append(
            {"role": "user", "content": user_content}
        )
        
        # Validate messages structure before processing
        logger.debug(f"Final messages structure: {len(messages)} messages")
        for i, msg in enumerate(messages):
            logger.debug(f"Message {i}: role={msg.get('role', 'MISSING')}, content_type={type(msg.get('content', 'MISSING'))}")

        inputs = selected_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=selected_model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(
            selected_processor, skip_prompt=True, skip_special_tokens=True, timeout=60.0
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
        
        # Wrapper function to catch thread exceptions
        def safe_generate():
            try:
                selected_model.generate(**generate_kwargs)
            except Exception as thread_e:
                logger.error(f"Exception in generation thread: {thread_e}")
                logger.error(f"Thread exception type: {type(thread_e)}")
                # Store the exception so we can handle it in the main thread
                import traceback
                logger.error(f"Thread traceback: {traceback.format_exc()}")
                raise
        
        t = Thread(target=safe_generate)
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
            selected_model, fallback_processor, fallback_name = try_fallback_model(model_choice)
            logger.info(f"Switching to fallback model: {fallback_name}")
            
            # Rebuild inputs for fallback model
            inputs = fallback_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device=selected_model.device, dtype=torch.bfloat16)

            streamer = TextIteratorStreamer(
                fallback_processor, skip_prompt=True, skip_special_tokens=True, timeout=60.0
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
            
            # Wrapper function to catch thread exceptions in fallback
            def safe_fallback_generate():
                try:
                    selected_model.generate(**generate_kwargs)
                except Exception as thread_e:
                    logger.error(f"Exception in fallback generation thread: {thread_e}")
                    logger.error(f"Fallback thread exception type: {type(thread_e)}")
                    import traceback
                    logger.error(f"Fallback thread traceback: {traceback.format_exc()}")
                    raise
            
            t = Thread(target=safe_fallback_generate)
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
