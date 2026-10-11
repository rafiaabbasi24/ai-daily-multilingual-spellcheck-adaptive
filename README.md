# Multilingual Contextual Spellchecker

## Project Overview

The **Multilingual Contextual Spellchecker** is a Python library that provides state‑of‑the‑art, context‑aware spelling correction for multiple languages. Leveraging transformer‑based language models and subword tokenization (via SentencePiece), the library can detect and correct spelling errors in sentences while preserving the original meaning and fluency.

Key features:

- **Multilingual support** – out‑of‑the‑box models for dozens of languages.
- **Contextual correction** – uses transformer encoders to consider surrounding words.
- **Fast inference** – optimized for CPU and GPU execution.
- **CLI interface** – quick command‑line usage with `click`.
- **Tested and documented** – comprehensive test suite with `pytest`.

## Tech Stack

- **Python** (≥3.8)
- **transformers** – Hugging Face 🤗 Transformers library for model loading.
- **sentencepiece** – Subword tokenizer.
- **click** – Command‑line interface.
- **numpy** – Numerical utilities.
- **pytest** – Unit testing framework.

## Installation

### Prerequisites

- Python 3.8 or newer
- pip (≥21.0)
- (Optional) CUDA toolkit for GPU acceleration

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-contextual-spellchecker.git
cd multilingual-contextual-spellchecker

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and its dependencies
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to modify the source code without reinstalling.

#### Installing optional GPU support

If you have a compatible GPU and want to use CUDA, install the PyTorch version that matches your CUDA toolkit:

```bash
# Example for CUDA 11.8
pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Quick Start

### Using the Python API

```python
from spellchecker import ContextualSpellChecker

# Load a pre‑trained multilingual model (e.g., "xlm-roberta-base")
checker = ContextualSpellChecker(model_name="xlm-roberta-base")

text = "I love progamming in Pythn."
corrected = checker.correct(text, language="en")
print(corrected)   # -> "I love programming in Python."
```

### Using the Command‑Line Interface

The package ships with a `spellcheck` CLI powered by `click`.

```bash
# Basic usage
spellcheck "Ths is a smple sentnce." --lang en

# Read from a file (one sentence per line)
spellcheck --input sentences.txt --output corrected.txt --lang fr

# Use a custom model
spellcheck "Ein Beispil satz." --model bert-base-multilingual-cased --lang de
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--model TEXT` | Name or path of the Hugging Face model to load (default: `xlm-roberta-base`). |
| `--lang TEXT` | ISO‑639‑1 language code (e.g., `en`, `fr`, `de`). |
| `--input FILE` | Path to a text file containing one sentence per line. |
| `--output FILE` | Path where corrected sentences will be written. |
| `--batch-size INTEGER` | Number of sentences processed per forward pass (default: 8). |
| `--device TEXT` | `cpu` or `cuda` (auto‑detects if omitted). |
| `--threshold FLOAT` | Confidence threshold for applying a correction (default: 0.8). |

## Development

### Running the Test Suite

The project uses `pytest`. To run all tests:

```bash
pytest -v
```

### Code Style

The code follows PEP 8 conventions and is type‑annotated. You can check formatting with `ruff` (or `flake8`) and type hints with `mypy`:

```bash
pip install ruff mypy
ruff check .
mypy src/
```

### Adding a New Language

1. **Create a tokenizer** (if not already covered by the base model) using `sentencepiece`:
   ```bash
   spm_train --input data/<lang>_corpus.txt --model_prefix <lang>_spm --vocab_size 32000
   ```
2. **Fine‑tune a transformer** on a masked language modeling (MLM) objective with your language data.
3. **Register the model** in `spellchecker/models.yaml` with its language code.
4. **Update the CLI help** to list the new language.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Write tests for your changes.
4. Ensure all tests pass (`pytest`).
5. Submit a pull request with a clear description of the changes.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgements

- Hugging Face 🤗 for the Transformers library.
- Google’s SentencePiece for subword tokenization.
- The open‑source community for numerous language resources.