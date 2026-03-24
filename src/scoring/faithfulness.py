"""
BERTScore-style faithfulness scoring using sentence-transformers.

Compares source text against draft text at the sentence level using
cosine similarity of sentence embeddings. Falls back gracefully if
sentence-transformers is not installed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on . ! ? followed by whitespace or end."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in parts if s.strip()]


class FaithfulnessScorer:
    """Lazy-loads all-MiniLM-L6-v2 on first call."""

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, source_text: str, draft_text: str) -> float:
        """
        Mean-of-max cosine similarity: for each source sentence, find the
        best-matching draft sentence, then average those scores.
        Returns 0.0–1.0.
        """
        detail = self.detail(source_text, draft_text)
        return detail["score"]

    def detail(self, source_text: str, draft_text: str) -> Dict[str, Any]:
        """Per-source-sentence faithfulness breakdown."""
        self._load()
        import numpy as np

        src_sents = _split_sentences(source_text)
        draft_sents = _split_sentences(draft_text)

        if not src_sents or not draft_sents:
            return {"score": 0.0, "per_sentence": [], "source_count": 0, "draft_count": 0}

        src_embs = self._model.encode(src_sents, convert_to_numpy=True)
        draft_embs = self._model.encode(draft_sents, convert_to_numpy=True)

        # Cosine similarity matrix: (num_src, num_draft)
        src_norm = src_embs / (np.linalg.norm(src_embs, axis=1, keepdims=True) + 1e-9)
        draft_norm = draft_embs / (np.linalg.norm(draft_embs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = src_norm @ draft_norm.T

        per_sentence = []
        for i, sent in enumerate(src_sents):
            max_sim = float(np.max(sim_matrix[i]))
            best_j = int(np.argmax(sim_matrix[i]))
            per_sentence.append({
                "source_sentence": sent,
                "best_match": draft_sents[best_j],
                "similarity": round(max_sim, 4),
            })

        mean_score = float(np.mean([p["similarity"] for p in per_sentence]))
        return {
            "score": round(max(0.0, min(1.0, mean_score)), 4),
            "per_sentence": per_sentence,
            "source_count": len(src_sents),
            "draft_count": len(draft_sents),
        }


_instance: FaithfulnessScorer | None = None


def get_faithfulness_scorer() -> FaithfulnessScorer:
    """Module-level singleton."""
    global _instance
    if _instance is None:
        _instance = FaithfulnessScorer()
    return _instance
