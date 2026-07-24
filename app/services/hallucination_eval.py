"""
Hallucination evaluation.

Compares a model's completion against a known reference answer to estimate
how "grounded" the completion is. If the completion shares little vocabulary
with the reference, it's more likely to contain unsupported/fabricated
content.

Contains score_hallucination() — compares the completion's words against a reference answer's words, returning a score from 0.0 (grounded) to 1.0 (likely hallucinated). If no reference is provided, returns a neutral 0.5.

Score convention: 0.0 = fully grounded, 1.0 = fully hallucinated.

Note: this is a simple lexical-overlap heuristic (Jaccard-style word
overlap), not a trained model. It's a reasonable first-pass signal and easy
to reason about, but it can't catch a confident-sounding fabrication that
happens to reuse the reference's wording, and it needs a reference answer
to compare against.
"""
import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def score_hallucination(completion: str, reference: str | None) -> float:
    """Returns a hallucination score between 0.0 (grounded) and 1.0 (likely
    hallucinated). If no reference answer is available, returns a neutral
    0.5 since there's nothing to check the completion against."""
    if not reference:
        return 0.5

    ref_tokens = _tokenize(reference)
    comp_tokens = _tokenize(completion)
    if not ref_tokens or not comp_tokens:
        return 0.5

    intersection = ref_tokens & comp_tokens
    overlap_ratio = len(intersection) / len(comp_tokens)
    # High overlap with the reference -> low hallucination score
    return round(max(0.0, 1.0 - overlap_ratio), 4)
