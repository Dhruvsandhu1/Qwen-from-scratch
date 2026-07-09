import torch
from src.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from transformers import AutoTokenizer

model_path='.'

# 1. (Optional) Load and inspect the explicit configuration
# This pulls the config.json and maps it to the Qwen2Config dataclass
config = Qwen2Config.from_json_file(f'{model_path}/config.json')
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Activation function: {config.hidden_act}")

# 2. Load the model directly into the Qwen2ForCausalLM class
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16, # Recommended precision for Qwen2.5
    device_map="auto"
)

# Note: It is still highly recommended to use AutoTokenizer because Qwen 
# relies on a fast tokenizer implementation written in Rust (tiktoken-based).
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. Setup the chat template
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write an essay on the importance of AI in modern education in 300 words."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. Tokenize and move to the same device as the model
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 5. Generate output
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7
    )

# 6. Isolate the new tokens and decode
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n--- Model Response ---")
print(response)