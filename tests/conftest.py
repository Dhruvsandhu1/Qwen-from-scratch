import os
import sys

# Make `src` and `triton_custom` importable when running `pytest tests` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: quick checks that the config and tokenizer load")
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (skipped otherwise)")
