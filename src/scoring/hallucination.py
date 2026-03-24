"""
NLI-based hallucination detection using a cross-encoder model.

For each sentence in the draft, classifies whether it is entailed by,
neutral to, or contradicts the source text. Sentences that are
contradictions or high-confidence neutral are flagged as potential
hallucinations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on . ! ? followed by whitespace or end."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in parts if s.strip()]


def _build_label_map(model: Any) -> Dict[int, str]:
    """
    Read the label order from the model's own config rather than hardcoding it.
    Different model revisions of nli-deberta-v3-xsmall ship different
    id-to-label mappings, so the only safe approach is to read it at runtime.
    Falls back to the most common ordering if the config isn't available.
    """
    config = getattr(model, "config", None) or getattr(model.model, "config", None)
    id2label = getattr(config, "id2label", None) if config else None
    if id2label and isinstance(id2label, dict):
        normalised: Dict[int, str] = {}
        for idx, raw_label in id2label.items():
            normalised[int(idx)] = str(raw_label).strip().lower()
        if {"entailment", "neutral", "contradiction"} <= set(normalised.values()):
            return normalised

    return {0: "contradiction", 1: "entailment", 2: "neutral"}


class HallucinationDetector:
    """Lazy-loads cross-encoder/nli-deberta-v3-xsmall on first call."""

    def __init__(self) -> None:
        self._model: Any = None
        self._label_map: Dict[int, str] = {}

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
            self._label_map = _build_label_map(self._model)

    def detect(
        self,
        source_text: str,
        draft_text: str,
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Classify each draft sentence against the full source text.

        Returns:
            flagged_sentences: list of sentences labeled contradiction or
                               high-confidence neutral
            hallucination_score: fraction of draft sentences flagged (0.0–1.0)
            details: per-sentence NLI breakdown
        """
        self._load()
        import numpy as np

        draft_sents = _split_sentences(draft_text)
        if not draft_sents or not (source_text or "").strip():
            return {"flagged_sentences": [], "hallucination_score": 0.0, "details": []}

        pairs = [(source_text.strip(), sent) for sent in draft_sents]
        scores = self._model.predict(pairs)

        details: List[Dict[str, Any]] = []
        flagged: List[str] = []

        for i, sent in enumerate(draft_sents):
            logits = scores[i]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            label_idx = int(np.argmax(probs))
            label = self._label_map.get(label_idx, "unknown")
            confidence = float(probs[label_idx])

            is_flagged = (
                label == "contradiction"
                or (label == "neutral" and confidence >= threshold)
            )
            if is_flagged:
                flagged.append(sent)

            details.append({
                "sentence": sent,
                "label": label,
                "confidence": round(confidence, 4),
                "flagged": is_flagged,
            })

        halluc_score = len(flagged) / len(draft_sents) if draft_sents else 0.0
        return {
            "flagged_sentences": flagged,
            "hallucination_score": round(halluc_score, 4),
            "details": details,
        }


_instance: HallucinationDetector | None = None


def get_hallucination_detector() -> HallucinationDetector:
    """Module-level singleton."""
    global _instance
    if _instance is None:
        _instance = HallucinationDetector()
    return _instance
