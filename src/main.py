"""Multilingual Contextual Spellchecker CLI.

This module provides a command‑line interface (CLI) built with Click that
allows users to run a simple contextual spell‑checking routine on a given
sentence.  The implementation leverages Hugging Face Transformers' *fill‑mask*
pipeline to generate correction candidates for each token in the input text.

Typical usage
-------------
>>> python -m src.main check --text "Ths is a smple sentence." --top-k 3

The command prints a table with the original token and the top‑k suggested
replacements together with their confidence scores.

The CLI is deliberately lightweight and does not aim to replace a full‑blown
spell‑checking system; it merely demonstrates how a multilingual masked language
model can be harnessed for contextual suggestions.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import List, Tuple

import click
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

# --------------------------------------------------------------------------- #
# Model handling
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_fill_mask_pipeline(model_name: str = "distilbert-base-multilingual-cased"):
    """Load a ``fill-mask`` pipeline for the given model.

    The function is cached so that the heavy model loading happens only once
    per interpreter session.

    Parameters
    ----------
    model_name: str
        Identifier of a Hugging Face model that supports masked language modeling.

    Returns
    -------
    transformers.pipelines.Pipeline
        A pipeline that accepts a string containing a ``[MASK]`` token and
        returns a list of candidate token predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)


def _mask_token(tokenizer) -> str:
    """Return the mask token string for a given tokenizer.

    Different models use different mask token representations (e.g. ``[MASK]``,
    ``<mask>``, ``<unk>``).  This helper abstracts that detail.
    """
    return tokenizer.mask_token


# --------------------------------------------------------------------------- #
# Spell‑checking core logic
# --------------------------------------------------------------------------- #

def suggest_corrections(
    text: str,
    top_k: int = 5,
    model_name: str = "distilbert-base-multilingual-cased",
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Generate correction suggestions for each token in *text*.

    For every whitespace‑separated token the function creates a masked version
    of the original sentence, runs the fill‑mask pipeline and extracts the
    *top_k* predictions.  The original token is paired with a list of tuples
    ``(suggestion, score)`` where *score* is the model's confidence.

    Parameters
    ----------
    text: str
        Input sentence to be analysed.
    top_k: int, optional
        Number of candidate suggestions to return per token (default: 5).
    model_name: str, optional
        Hugging Face model identifier (default: multilingual DistilBERT).

    Returns
    -------
    List[Tuple[str, List[Tuple[str, float]]]]
        A list where each element corresponds to a token in the original text.
        The first item of the tuple is the original token; the second item is a
        list of ``(suggestion, score)`` pairs ordered by descending confidence.
    """
    if not text:
        raise ValueError("Input text must be a non‑empty string.")

    # Load pipeline and tokenizer once
    fill_mask = _load_fill_mask_pipeline(model_name)
    tokenizer = fill_mask.tokenizer
    mask = _mask_token(tokenizer)

    tokens = text.split()
    results: List[Tuple[str, List[Tuple[str, float]]]] = []

    for idx, token in enumerate(tokens):
        # Build a masked sentence where only the current token is replaced.
        masked_tokens = tokens.copy()
        masked_tokens[idx] = mask
        masked_sentence = " ".join(masked_tokens)

        # Run the pipeline; it returns a list of dictionaries.
        try:
            predictions = fill_mask(masked_sentence, top_k=top_k)
        except Exception as exc:
            click.echo(f"Warning: model inference failed for token '{token}': {exc}", err=True)
            results.append((token, []))
            continue

        # Normalise scores and collect suggestions.
        suggestions = [
            (pred["token_str"].strip(), float(pred["score"])) for pred in predictions
        ]
        results.append((token, suggestions))

    return results


# --------------------------------------------------------------------------- #
# Click command‑line interface
# --------------------------------------------------------------------------- #

@click.group()
def cli() -> None:
    """Multilingual Contextual Spellchecker command line interface."""
    pass


@cli.command()
@click.option(
    "--text",
    "-t",
    required=True,
    type=str,
    help="Sentence to be spell‑checked.",
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of suggestions per token.",
)
@click.option(
    "--model",
    "-m",
    default="distilbert-base-multilingual-cased",
    show_default=True,
    type=str,
    help="Hugging Face model name for the fill‑mask pipeline.",
)
def check(text: str, top_k: int, model: str) -> None:
    """Run the contextual spell‑checker on *TEXT* and display suggestions."""
    click.echo(f"Running spell‑check on: {text!r}")
    try:
        suggestions = suggest_corrections(text, top_k=top_k, model_name=model)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not suggestions:
        click.echo("No tokens to process.")
        return

    # Pretty‑print the results as a table.
    header = f"{'Token':<20} | {'Suggestions (score)':<50}"
    click.echo("-" * len(header))
    click.echo(header)
    click.echo("-" * len(header))

    for original, cand_list in suggestions:
        if not cand_list:
            line = f"{original:<20} | (no suggestions)"
        else:
            formatted = ", ".join(f"{s} ({score:.2f})" for s, score in cand_list)
            line = f"{original:<20} | {formatted:<50}"
        click.echo(line)

    click.echo("-" * len(header))


if __name__ == "__main__":
    cli()