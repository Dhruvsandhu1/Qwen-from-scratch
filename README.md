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

## Notes
- Large model files (`*.safetensors`) are large and often excluded from VCS; see `.gitignore`.
- If you obtained model files from a third-party provider, follow their licensing and usage terms.

## Development
- Code lives in `src/` and follows a standard Python package layout.
- Run linters and formatters as preferred (e.g., `black`, `flake8`).

## License
This project is released under the MIT License — see `LICENSE`.
