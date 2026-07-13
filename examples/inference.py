"""Inference script for the Qwen2 model"""

import torch

from src.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoConfig
import argparse
import sys
from loguru import logger

# configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} {level} {message}")

parser = argparse.ArgumentParser(
    prog="Qwen2 Inference Script",
    description="This script demonstrates how to perform inference using the Qwen2 model for causal language modeling.",
    epilog="Example usage: python inference.py --model_path ./model",
)
parser.add_argument("--model_path", type=str, default="./model", help="Path to the pretrained model")
parser.add_argument(
    "--input_text",
    type=str,
    default="Write an essay on the importance of AI in modern education in 300 words.",
    help="Input text for the model to generate a response",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    help="Data type for model weights (e.g., bfloat16, float32)",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=1024,
    help="Maximum number of new tokens to generate",
)
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
parser.add_argument(
    "--device_map",
    type=str,
    default="cuda",
    help='Device map for from_pretrained (e.g., "auto", "cpu", or a mapping)',
)
args = parser.parse_args()

model_path = args.model_path

# 1. (Optional) Load and inspect the explicit configuration
config = AutoConfig.from_pretrained(model_path)
logger.info(f"Number of layers: {config.num_hidden_layers}")
logger.info(f"Hidden size: {config.hidden_size}")
logger.info(f"Activation function: {config.hidden_act}")

# Validate the user input
if not isinstance(getattr(torch, args.dtype, None), torch.dtype):
    raise ValueError(f"Invalid dtype: '{args.dtype}'")

# 2. Load the model directly into the Qwen2ForCausalLM class
try:
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map if args.device_map != "cpu" else None,
        attn_implementation="flash_attention_2",
    )
except Exception as e:
    logger.exception("Failed to load model: {}", e)
    raise

# Note: It is still highly recommended to use AutoTokenizer because Qwen
# relies on a fast tokenizer implementation written in Rust (tiktoken-based).
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. Setup the chat template
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": args.input_text},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Tokenize and move to the same device as the model
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 5. Generate output
with torch.no_grad():
    generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

# 6. Isolate the new tokens and decode
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

logger.info("\n--- Model Response ---")
logger.info(response)
