# Qwen2.5 — Local Inference Project

This repository contains code and tokenizer/model artifacts to run local inference for the Qwen2.5 model.

## Contents
- `inference.py` — example runner for model inference.
- `src/` — model configuration, tokenizer and modeling code.
- Model files (`*.safetensors`, indexes) — large artifacts are typically not committed.

## Quickstart
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run inference (adjust arguments as needed):

```bash
python inference.py
```

Interactive mode
----------------
To run an interactive session that keeps the model loaded between prompts:

```bash
python examples/interactive_inference.py --model_path model
```

Press Ctrl-C or type `exit` to quit. After each response press Enter to clear the screen and continue.

Editable install
----------------
Install the package in editable mode so you can import `src` and edit code in-place:

```bash
pip install -e .
```

Testing
-------
Run the smoke tests (they skip if `model/` is not present):

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes
- Large model files (`*.safetensors`) are large and often excluded from VCS; see `.gitignore`.
- If you obtained model files from a third-party provider, follow their licensing and usage terms.

## Development
- Code lives in `src/` and follows a standard Python package layout.
- Run linters and formatters as preferred (e.g., `black`, `flake8`).

## License
This project is released under the MIT License — see `LICENSE`.
