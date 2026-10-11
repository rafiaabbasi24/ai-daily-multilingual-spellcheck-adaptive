"""tests/test_spellchecker.py
Unit tests for the multilingual contextual spellchecker.

These tests verify that the spellchecker can detect misspelled tokens and
produce corrected output for several languages.  The tests use a tiny
multilingual model to keep execution time low while still exercising the
full detection/correction pipeline.

The public API expected from the package is:
    - ``ContextualSpellChecker`` class in ``multilingual_spellchecker``.
    - ``detect(text: str) -> List[Tuple[int, int, str]]`` returning a list of
      (start_char, end_char, misspelled_token) tuples.
    - ``correct(text: str) -> str`` returning the corrected string.

If the real implementation differs, adjust the imports / method calls
accordingly.
"""

from __future__ import annotations

import os
import pytest

# The package under test.  Adjust the import path if the project uses a different
# module name.
from multilingual_spellchecker import ContextualSpellChecker

# Use a tiny multilingual model to keep the test fast and deterministic.
# ``sshleifer/tiny-multilingual-bert`` is a publicly available model on
# HuggingFace that works for English, Spanish and French.
_MODEL_NAME = "sshleifer/tiny-multilingual-bert"


@pytest.fixture(scope="module")
def spellchecker() -> ContextualSpellChecker:
    """Create a spellchecker instance for the whole test module.

    The fixture loads the model once and re‑uses it across tests.
    """
    # Ensure transformers does not try to use GPU in CI environments.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Instantiate the spellchecker.  The constructor signature may accept a
    # ``model_name`` argument; adjust if the actual API differs.
    return ContextualSpellChecker(model_name=_MODEL_NAME)


def test_detection_of_english_misspelling(spellchecker: ContextualSpellChecker) -> None:
    """The spellchecker should locate a simple English typo."""
    text = "I havve a dream."
    detections = spellchecker.detect(text)

    # Expect exactly one detection for the word "havve".
    assert len(detections) == 1, f"Expected 1 detection, got {len(detections)}"
    start, end, token = detections[0]
    assert token == "havve"
    assert text[start:end] == token


def test_correction_of_english_misspelling(spellchecker: ContextualSpellChecker) -> None:
    """The spellchecker should replace the typo with the correct word."""
    text = "I havve a dream."
    corrected = spellchecker.correct(text)
    assert corrected == "I have a dream.", f"Incorrect correction: {corrected}"


def test_detection_of_spanish_misspelling(spellchecker: ContextualSpellChecker) -> None:
    """The spellchecker must also work for Spanish."""
    text = "Me gusta el chocolet."
    detections = spellchecker.detect(text)

    assert len(detections) == 1, f"Expected 1 detection, got {len(detections)}"
    _, _, token = detections[0]
    assert token == "chocolet"


def test_correction_of_spanish_misspelling(spellchecker: ContextualSpellChecker) -> None:
    """The corrected Spanish sentence should contain the proper word."""
    text = "Me gusta el chocolet."
    corrected = spellchecker.correct(text)
    # The model may keep punctuation; we only check the corrected token.
    assert "chocolate" in corrected, f"Spanish correction missing: {corrected}"


def test_no_false_positives_on_correct_sentence(spellchecker: ContextualSpellChecker) -> None:
    """A sentence without errors should yield an empty detection list."""
    text = "The quick brown fox jumps over the lazy dog."
    detections = spellchecker.detect(text)
    assert detections == [], f"Unexpected detections on a correct sentence: {detections}"


def test_correction_preserves_correct_words(spellchecker: ContextualSpellChecker) -> None:
    """Correct sentences must remain unchanged after correction."""
    text = "The quick brown fox jumps over the lazy dog."
    corrected = spellchecker.correct(text)
    assert corrected == text, "Correct sentence was altered by the spellchecker"


def test_multilingual_correction_french(spellchecker: ContextualSpellChecker) -> None:
    """Validate correction on a simple French typo."""
    text = "J'aime le chôcolat noir."
    corrected = spellchecker.correct(text)
    # The expected correction is "chocolat".  We allow surrounding context to stay.
    assert "chocolat" in corrected, f"French correction missing: {corrected}"
    # Ensure the accent on "chôcolat" is removed and the word is spelled correctly.
    assert "chôcolat" not in corrected, "Accent typo was not corrected"


def test_detection_returns_character_offsets(spellchecker: ContextualSpellChecker) -> None:
    """Offsets returned by ``detect`` must correspond to the original text."""
    text = "She recieved the package."
    detections = spellchecker.detect(text)
    assert len(detections) == 1
    start, end, token = detections[0]
    # The misspelled token is "recieved".
    assert token == "recieved"
    # Verify that slicing the original text reproduces the token.
    assert text[start:end] == token
    # Ensure the offsets are within the string bounds.
    assert 0 <= start < len(text)
    assert start < end <= len(text)