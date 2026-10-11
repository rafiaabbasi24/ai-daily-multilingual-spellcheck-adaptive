import os
import re
import unicodedata
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
import torch
from joblib import Memory
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# --------------------------------------------------------------------------- #
# Caching utilities
# --------------------------------------------------------------------------- #

# A disk‑based cache directory placed next to the source tree.
# The directory is created lazily on first use.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_MEMORY = Memory(location=str(_CACHE_DIR), verbose=0)


def disk_cache(func: Callable) -> Callable:
    """
    Decorator that caches the result of ``func`` on disk using ``joblib.Memory``.
    The cache key is built from the function arguments (pickled).  This is
    useful for expensive operations such as model loading or large preprocessing
    steps that are deterministic.

    Parameters
    ----------
    func: Callable
        The function to be cached.

    Returns
    -------
    Callable
        Wrapped function with transparent disk caching.
    """
    cached_func = _MEMORY.cache(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return cached_func(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #


def get_device(preferred: str = "cpu") -> torch.device:
    """
    Resolve a ``torch.device`` based on user preference and hardware availability.

    Parameters
    ----------
    preferred : str, optional
        Desired device name (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).  If the
        preferred device is unavailable, the function falls back to ``cpu``.

    Returns
    -------
    torch.device
        The device that should be used for model inference.
    """
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred.lower() == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


@disk_cache
def _load_tokenizer(model_name_or_path: str, **kwargs) -> PreTrainedTokenizer:
    """
    Load a tokenizer from the HuggingFace hub or a local directory.

    This function is cached on disk because tokenizer loading can be slow
    for large vocabularies.

    Parameters
    ----------
    model_name_or_path : str
        Identifier of the model (e.g., ``"facebook/mbart-large-50"``) or a local
        path containing the tokenizer files.
    **kwargs
        Additional arguments passed to ``AutoTokenizer.from_pretrained``.

    Returns
    -------
    PreTrainedTokenizer
        The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


@disk_cache
def _load_model(
    model_name_or_path: str,
    model_type: str = "seq2seq",
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> PreTrainedModel:
    """
    Load a transformer model from the HuggingFace hub or a local directory.

    The model is loaded onto the requested device and cached on disk to avoid
    repeated downloads.

    Parameters
    ----------
    model_name_or_path : str
        Identifier of the model or a local directory.
    model_type : {"seq2seq", "masked_lm", "auto"}, optional
        Determines which ``AutoModel`` class to instantiate.
    device : str or torch.device, optional
        Device where the model will be placed.
    **kwargs
        Additional arguments forwarded to the ``from_pretrained`` call.

    Returns
    -------
    PreTrainedModel
        The model ready for inference.
    """
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
    elif model_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_name_or_path, **kwargs)

    device = get_device(str(device))
    model.to(device)
    model.eval()
    return model


def load_tokenizer_and_model(
    model_name_or_path: str,
    model_type: str = "seq2seq",
    device: Union[str, torch.device] = "cpu",
    cache_dir: Union[str, os.PathLike, None] = None,
    **kwargs,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Convenience wrapper that returns both tokenizer and model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
    model_type : {"seq2seq", "masked_lm", "auto"}, optional
        Which model class to use.
    device : str or torch.device, optional
        Target device for the model.
    cache_dir : str or Path, optional
        Directory used by ``transformers`` for caching downloaded files.
        If ``None`` the default cache location is used.
    **kwargs
        Extra arguments forwarded to the underlying ``from_pretrained`` calls.

    Returns
    -------
    tuple(PreTrainedTokenizer, PreTrainedModel)
        Tokenizer and model ready for inference.
    """
    extra = {"cache_dir": str(cache_dir)} if cache_dir else {}
    tokenizer = _load_tokenizer(model_name_or_path, **extra)
    model = _load_model(model_name_or_path, model_type=model_type, device=device, **extra)
    return tokenizer, model


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #


_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_CHARS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE
}


def _remove_zero_width(text: str) -> str:
    """Remove invisible zero‑width characters that can break tokenisation."""
    return "".join(ch for ch in text if ch not in _ZERO_WIDTH_CHARS)


def normalize_text(
    text: str,
    lower: bool = True,
    normalize_unicode: bool = True,
    strip: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """
    Apply a series of deterministic normalisation steps to raw input text.

    The function is deliberately lightweight and does not perform language‑specific
    processing; it is suitable for multilingual pipelines.

    Parameters
    ----------
    text : str
        Raw string to be cleaned.
    lower : bool, optional
        Convert the text to lower case.
    normalize_unicode : bool, optional
        Apply Unicode NFKC normalisation (recommended for most NLP tasks).
    strip : bool, optional
        Remove leading and trailing whitespace.
    collapse_whitespace : bool, optional
        Replace consecutive whitespace characters with a single space.

    Returns
    -------
    str
        Normalised string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected `text` to be str, got {type(text)!r}")

    # Remove zero‑width characters first – they can interfere with later steps.
    text = _remove_zero_width(text)

    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    if lower:
        text = text.lower()

    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)

    if strip:
        text = text.strip()

    return text


def batch_normalize(
    texts: Iterable[str],
    **kwargs,
) -> List[str]:
    """
    Apply :func:`normalize_text` to an iterable of strings.

    Parameters
    ----------
    texts : Iterable[str]
        Collection of raw texts.
    **kwargs
        Keyword arguments forwarded to :func:`normalize_text`.

    Returns
    -------
    List[str]
        List of normalised strings preserving the original order.
    """
    return [normalize_text(t, **kwargs) for t in texts]


# --------------------------------------------------------------------------- #
# Embedding utilities (optional but handy for spell‑checking contexts)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def _get_embedding_layer(model: PreTrainedModel) -> torch.nn.Module:
    """
    Heuristically retrieve the hidden‑state output layer from a model.
    This works for most encoder‑only and encoder‑decoder architectures.

    The result is cached because the lookup is performed repeatedly during
    batch inference.
    """
    # Common attribute names across HuggingFace models
    for attr in ("encoder", "model", "transformer"):
        if hasattr(model, attr):
            submodule = getattr(model, attr)
            if hasattr(submodule, "last_hidden_state"):
                return submodule
            # For encoder‑decoder models the encoder itself may expose
            # ``hidden_states`` directly.
            if hasattr(submodule, "hidden_states"):
                return submodule
    # Fallback: the model itself should produce hidden states when called
    return model


def get_sentence_embeddings(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute mean‑pooled sentence embeddings for a list of texts.

    The function tokenises the inputs, runs them through the model,
    extracts the last hidden state, and averages over the token dimension
    (ignoring padding tokens).

    Parameters
    ----------
    texts : List[str]
        Sentences to embed.
    tokenizer : PreTrainedTokenizer
        Tokenizer compatible with ``model``.
    model : PreTrainedModel
        Transformer model that returns hidden states.
    device : str or torch.device, optional
        Device for inference.
    batch_size : int, optional
        Number of sentences processed simultaneously.
    max_length : int, optional
        Maximum token length; longer texts are truncated.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), hidden_dim)`` containing the embeddings.
    """
    device = get_device(str(device))
    model.to(device)

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            # Prefer the last hidden state; if not present, fall back to hidden_states[-1]
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]

        # Create mask for non‑padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        masked_hidden = hidden * attention_mask

        # Mean‑pool while ignoring padding
        summed = masked_hidden.sum(dim=1)  # (B, D)
        lengths = attention_mask.sum(dim=1)  # (B, 1)
        batch_emb = summed / torch.clamp(lengths, min=1.0)

        all_embeddings.append(batch_emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "get_device",
    "load_tokenizer_and_model",
    "normalize_text",
    "batch_normalize",
    "get_sentence_embeddings",
    "disk_cache",
]