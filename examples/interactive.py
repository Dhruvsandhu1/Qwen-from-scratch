"""Interactive inference CLI for Qwen2 models.

Loads the model once, then accepts user input in a loop. After showing
the response the screen is cleared (so the next user sees a clean prompt),
but the model remains loaded in memory for subsequent queries.
"""

import os
import sys
import argparse
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoConfig
from src.modeling_qwen2 import Qwen2ForCausalLM


def clear_screen() -> None:
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def main():
    parser = argparse.ArgumentParser(description="Interactive Qwen2 inference")
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

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} {level} {message}")

    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path)
    logger.info("Loaded config: model_type=%s, layers=%s", config.model_type, config.num_hidden_layers)

    if not isinstance(getattr(torch, args.dtype, None), torch.dtype):
        raise ValueError(f"Invalid dtype: '{args.dtype}'")
    torch_dtype = getattr(torch, args.dtype)

    logger.info("Loading model (this may take a while)...")
    model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch_dtype, device_map=args.device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Model loaded. Enter prompts (type 'exit' or Ctrl-C to quit).")

    try:
        while True:
            prompt = input("You: ")
            if not prompt:
                continue
            if prompt.strip().lower() in ("exit", "quit"):
                break

            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

            generated_ids = [out[len(inp) :] for inp, out in zip(model_inputs.input_ids, generated)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print("\nAssistant:\n")
            print(response)

            input("\nPress Enter to clear the screen and continue...")
            clear_screen()

    except KeyboardInterrupt:
        logger.info("Exiting interactive session.")


if __name__ == "__main__":
    main()
