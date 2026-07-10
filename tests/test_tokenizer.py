import os
import pytest
from transformers import AutoTokenizer


@pytest.mark.smoke
def test_tokenizer_encode_decode():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        pytest.skip(f"Model directory not found at {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = "Hello, world!"
    encoded = tokenizer(sample)
    assert "input_ids" in encoded
    decoded = (
        tokenizer.decode(encoded["input_ids"])
        if isinstance(encoded["input_ids"], list)
        else tokenizer.decode(encoded.input_ids[0])
    )
    assert isinstance(decoded, str)
