import sys
from pathlib import Path
from typing import List, Tuple, Optional

import click
import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)

# Global caches to avoid reloading models multiple times
_LANGUAGE_DETECTOR = None
_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None
_MLM_MODEL = None
_MLM_TOKENIZER = None


def _load_language_detector():
    """Load a lightweight language detection pipeline.

    Uses the HuggingFace model ``papluca/xlm-roberta-base-language-detection``.
    The pipeline returns a list of dictionaries with ``label`` and ``score``.
    """
    global _LANGUAGE_DETECTOR
    if _LANGUAGE_DETECTOR is None:
        _LANGUAGE_DETECTOR = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            tokenizer="papluca/xlm-roberta-base-language-detection",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _LANGUAGE_DETECTOR


def _load_embedding_model():
    """Load a multilingual sentence embedding model.

    The model ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
    provides 384‑dimensional embeddings suitable for similarity calculations.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER
    if _EMBEDDING_MODEL is None or _EMBEDDING_TOKENIZER is None:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _EMBEDDING_MODEL = AutoModel.from_pretrained(model_name)
        _EMBEDDING_MODEL.eval()
        if torch.cuda.is_available():
            _EMBEDDING_MODEL.to("cuda")
    return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER


def _load_mlm_model():
    """Load a multilingual masked language model for candidate generation.

    The model ``bert-base-multilingual-cased`` is a widely used MLM that
    supports over 100 languages.
    """
    global _MLM_MODEL, _MLM_TOKENIZER
    if _MLM_MODEL is None or _MLM_TOKENIZER is None:
        model_name = "bert-base-multilingual-cased"
        _MLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MLM_MODEL = AutoModelForMaskedLM.from_pretrained(model_name)
        _MLM_MODEL.eval()
        if torch.cuda.is_available():
            _MLM_MODEL.to("cuda")
    return _MLM_MODEL, _MLM_TOKENIZER


def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on token embeddings.

    Args:
        model_output: Last hidden states from the transformer (batch, seq_len, hidden).
        attention_mask: Binary mask indicating real tokens (batch, seq_len).

    Returns:
        Tensor of shape (batch, hidden) containing the mean‑pooled embeddings.
    """
    token_embeddings = model_output[0]  # First element is the hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class SpellChecker:
    """
    Multilingual contextual spellchecker.

    The workflow consists of:
    1. Detecting the language of the input text.
    2. Computing a context embedding for the whole sentence.
    3. Generating candidate corrections for each potentially misspelled token.
    4. Ranking candidates using cosine similarity with the context embedding.
    """

    def __init__(self, top_k: int = 5):
        """
        Initialise the spellchecker.

        Args:
            top_k: Number of candidate corrections to return per token.
        """
        self.top_k = top_k
        self.language_detector = _load_language_detector()
        self.embed_model, self.embed_tokenizer = _load_embedding_model()
        self.mlm_model, self.mlm_tokenizer = _load_mlm_model()

    # --------------------------------------------------------------------- #
    # Language detection
    # --------------------------------------------------------------------- #
    def detect_language(self, text: str) -> str:
        """
        Detect the language of ``text`` using a pretrained classifier.

        Returns:
            ISO‑639‑1 language code (e.g., ``en``, ``fr``).
        """
        results = self.language_detector(text)
        # The pipeline returns a list with a single dict for single‑sentence input
        best = max(results, key=lambda x: x["score"])
        # Labels are of the form ``en`` or ``en_XX`` – we keep the first two chars
        return best["label"][:2]

    # --------------------------------------------------------------------- #
    # Embedding utilities
    # --------------------------------------------------------------------- #
    def _encode(self, text: str) -> np.ndarray:
        """
        Encode ``text`` into a dense vector using the multilingual sentence model.

        Returns:
            1‑D numpy array of dtype ``float32``.
        """
        inputs = self.embed_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            model_output = self.embed_model(**inputs)
        embedding = _mean_pooling(model_output, inputs["attention_mask"])
        return embedding.cpu().numpy().flatten().astype(np.float32)

    # --------------------------------------------------------------------- #
    # Candidate generation
    # --------------------------------------------------------------------- #
    def generate_candidates(self, sentence: str, token_idx: int) -> List[Tuple[str, float]]:
        """
        Generate candidate replacements for the token at ``token_idx`` in ``sentence``.

        The method masks the target token, runs the MLM, and returns the top‑k
        predictions together with their softmax probabilities.

        Args:
            sentence: Full input sentence.
            token_idx: Index of the token (0‑based) after whitespace tokenisation.

        Returns:
            List of ``(candidate, probability)`` tuples ordered by decreasing probability.
        """
        tokens = self.mlm_tokenizer.tokenize(sentence)
        # Align whitespace tokenisation with BERT tokenisation.
        # We locate the start token index that corresponds to ``token_idx``.
        # Simple heuristic: split on whitespace and map each word to its first sub‑token.
        words = sentence.split()
        if token_idx >= len(words):
            raise IndexError("token_idx out of range for the provided sentence.")
        target_word = words[token_idx]

        # Find the position of the first sub‑token that matches the target word.
        # This works for most cases; for complex cases (e.g., punctuation) we fall back to masking the whole word.
        sub_token_start = None
        accumulated = ""
        for i, sub in enumerate(tokens):
            accumulated += sub.replace("##", "")
            if accumulated == target_word:
                sub_token_start = i
                break
            if len(accumulated) > len(target_word):
                break
        if sub_token_start is None:
            # Fallback: mask the whole word by locating its character span.
            char_start = sentence.find(target_word)
            char_end = char_start + len(target_word)
            masked_sentence = (
                sentence[:char_start] + self.mlm_tokenizer.mask_token + sentence[char_end:]
            )
        else:
            # Build masked token list
            masked_tokens = tokens.copy()
            masked_tokens[sub_token_start] = self.mlm_tokenizer.mask_token
            masked_sentence = self.mlm_tokenizer.convert_tokens_to_string(masked_tokens)

        inputs = self.mlm_tokenizer(
            masked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        mask_position = (inputs["input_ids"] == self.mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.mlm_model(**inputs).logits
        mask_logits = logits[0, mask_position, :]
        probs = torch.softmax(mask_logits, dim=-1).squeeze()
        topk = torch.topk(probs, self.top_k)
        candidate_ids = topk.indices.tolist()
        candidate_probs = topk.values.tolist()
        candidates = [
            (self.mlm_tokenizer.decode([cid]).strip(), prob) for cid, prob in zip(candidate_ids, candidate_probs)
        ]
        return candidates

    # --------------------------------------------------------------------- #
    # Ranking
    # --------------------------------------------------------------------- #
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity between two 1‑D arrays."""
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def rank_candidates(
        self,
        context_embedding: np.ndarray,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate words by combining MLM probability with contextual similarity.

        The final score is ``0.5 * prob + 0.5 * similarity`` (both normalized to [0,1]).

        Returns:
            List of ``(candidate, final_score)`` sorted descending.
        """
        # Normalise probabilities
        probs = np.array([p for _, p in candidates], dtype=np.float32)
        prob_norm = (probs - probs.min()) / (probs.ptp() + 1e-9)

        # Compute similarity for each candidate
        sims = []
        for word, _ in candidates:
            cand_emb = self._encode(word)
            sims.append(self._cosine_similarity(context_embedding, cand_emb))
        sims = np.array(sims, dtype=np.float32)
        sim_norm = (sims - sims.min()) / (sims.ptp() + 1e-9)

        final_scores = 0.5 * prob_norm + 0.5 * sim_norm
        ranked = sorted(
            zip([c for c, _ in candidates], final_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def correct_sentence(self, sentence: str) -> str:
        """
        Detect misspelled tokens and return a corrected sentence.

        This simplistic implementation treats every token as a potential error,
        generates candidates, ranks them, and substitutes the top candidate if its
        score exceeds a small threshold (0.6). In practice, a separate error
        detection model would be used.

        Args:
            sentence: Input sentence.

        Returns:
            Corrected sentence string.
        """
        words = sentence.split()
        context_emb = self._encode(sentence)

        corrected_words = words.copy()
        for idx, _ in enumerate(words):
            candidates = self.generate_candidates(sentence, idx)
            ranked = self.rank_candidates(context_emb, candidates)
            best_word, best_score = ranked[0]
            if best_score > 0.6 and best_word != words[idx]:
                corrected_words[idx] = best_word

        return " ".join(corrected_words)


# -------------------------------------------------------------------------
# CLI implementation
# -------------------------------------------------------------------------
@click.group()
def cli():
    """Multilingual Contextual Spellchecker command line interface."""
    pass


@cli.command()
@click.argument("text", type=str, required=False)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a text file containing a single sentence.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of candidate corrections to consider per token.",
)
def correct(text: Optional[str], file: Optional[str], top_k: int):
    """
    Correct a sentence supplied directly or via a file.

    Example:

        python -m src.spellchecker correct "Ths is a smple sentnce."

    Or:

        python -m src.spellchecker correct -f my_sentence.txt
    """
    if not text and not file:
        click.echo("Either TEXT argument or --file must be provided.", err=True)
        sys.exit(1)

    if file:
        txt_path = Path(file)
        text = txt_path.read_text(encoding="utf-8").strip()

    checker = SpellChecker(top_k=top_k)
    corrected = checker.correct_sentence(text)
    click.echo(corrected)


if __name__ == "__main__":
    cli()