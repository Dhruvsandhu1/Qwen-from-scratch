import os
import pytest
from transformers import AutoConfig


@pytest.mark.smoke
def test_config_load():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        pytest.skip(f"Model directory not found at {model_dir}")

    config = AutoConfig.from_pretrained(model_dir)
    # basic sanity checks
    assert hasattr(config, "model_type")
    assert getattr(config, "vocab_size", None) is not None
    assert getattr(config, "num_hidden_layers", None) is not None
