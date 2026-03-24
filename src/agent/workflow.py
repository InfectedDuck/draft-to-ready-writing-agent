import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from src.llm.mock_client import MockLLMClient
from src.prompts.templates import (
    CLARIFY_PROMPT_TEMPLATE,
    DRAFT_PROMPT_TEMPLATE,
    FINALIZE_PROMPT_TEMPLATE,
    JUDGE_DRAFT_PROMPT_TEMPLATE,
)
from src.schemas.models import DraftRequest, DraftResponse


def _load_scoring_weights() -> Dict[str, float]:
    """
    Optional weights tuning for the heuristic selector.
    If `evals/scoring_weights.json` exists, it overrides defaults.
    """
    defaults = {
        # Sorted by seriousness: content accuracy > tone/style > structure > formatting
        "w_faithfulness": 1.5,     # Most important: does the draft reflect the user's notes?
        "w_hallucination": 1.4,    # Critical: penalize fabricated details heavily
        "w_intent": 1.2,           # Important: are required details (dates, amounts) present?
        "w_tone": 1.1,             # Important: does it match the requested style preset?
        "w_next_step": 0.9,        # Medium: actionable language matching the preset
        "w_subject": 0.8,          # Medium: channel-appropriate subject line
        "w_word_size": 0.6,        # Lower: word count target is a soft guideline
        "w_closing": 0.5,          # Lower: sign-off is nice-to-have, not critical
        "w_short": 0.4,            # Lowest: minimum length is a basic sanity check
    }
    try:
        weights_path = Path(__file__).resolve().parents[2] / "evals" / "scoring_weights.json"
        if weights_path.exists():
            with weights_path.open("r", encoding="utf-8") as f:
                user_weights = json.load(f)
            for k, v in user_weights.items():
                if k in defaults and isinstance(v, (int, float)):
                    defaults[k] = float(v)
    except Exception:
        pass
    return defaults


SCORING_WEIGHTS = _load_scoring_weights()


def _ollama_options_from_req(req: DraftRequest) -> dict:
    opts = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repeat_penalty": req.repeat_penalty,
        "presence_penalty": req.presence_penalty,
        "frequency_penalty": req.frequency_penalty,
    }
    if req.seed is not None:
        opts["seed"] = req.seed
    return opts


def _word_count_block(req: DraftRequest) -> str:
    # Approximate word targets so the user doesn't need to provide exact counts.
    # These are broad ranges so models can naturally hit them.
    ranges = {
        "Small": (60, 90),
        "Medium": (110, 160),
        "Large": (180, 260),
    }
    lo, hi = ranges.get(req.word_size, (110, 160))
    return f"Aim for approximately {lo}-{hi} words."


def _basic_rubric_check(draft: str, req: Optional[DraftRequest] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Cheap self-check that doesn't need another LLM call.
    Returns a rubric dict with boolean pass/fail checks and a questions list.
    """
    rubric: Dict[str, Any] = {}
    questions: List[str] = []

    draft_lower = (draft or "").lower()
    wc = len((draft or "").split())

    # --- Core checks ---
    rubric["has_closing"] = any(
        x in draft_lower for x in ["sincerely", "regards", "thanks", "thank you", "yours", "best", "cheers", "warm regards"]
    )
    rubric["mentions_next_step"] = any(
        x in draft_lower for x in ["if you need", "please", "would", "can you", "i can", "provide", "let me know", "feel free"]
    )
    rubric["not_too_short"] = wc >= 40

    # --- Greeting check ---
    rubric["has_greeting"] = any(
        x in draft_lower for x in ["hi", "hello", "dear", "good morning", "good afternoon", "good evening", "hey"]
    )

    # --- Paragraph structure (not a wall of text) ---
    lines = [ln.strip() for ln in (draft or "").splitlines() if ln.strip()]
    rubric["has_paragraphs"] = len(lines) >= 3

    # --- Channel-specific checks ---
    if req:
        channel_l = (req.channel or "").lower()
        has_subject = "subject:" in draft_lower

        if channel_l == "email":
            rubric["email_has_subject"] = has_subject if req.include_subject else True
            rubric["email_has_greeting"] = rubric["has_greeting"]
        elif "whatsapp" in channel_l:
            rubric["whatsapp_no_subject"] = not has_subject
            rubric["whatsapp_concise"] = wc <= 120
        elif "teams" in channel_l:
            rubric["teams_no_subject"] = not has_subject
            rubric["teams_professional"] = rubric["has_greeting"] and rubric["has_closing"]

        # --- Word count in target range ---
        lo, hi = _target_word_range(req.word_size)
        rubric["word_count_in_range"] = lo <= wc <= hi
        rubric["word_count"] = wc
        rubric["word_count_target"] = f"{lo}-{hi}"

        # --- Tone markers present ---
        markers = _style_tone_markers_for_preset(req.style_preset)
        marker_hits = sum(1 for m in markers if m.lower() in draft_lower)
        rubric["tone_markers_found"] = marker_hits
        rubric["tone_markers_total"] = len(markers)
        rubric["tone_match"] = marker_hits >= 1  # at least one marker

    # Ask for missing info only when the draft seems too generic.
    if wc < 20:
        questions.append("Add more details (dates, deadlines, specific request) so the draft can be more concrete.")

    return rubric, questions


def _target_word_range(word_size: str) -> tuple[int, int]:
    if word_size == "Small":
        return (60, 90)
    if word_size == "Large":
        return (180, 260)
    return (110, 160)


def _count_words(text: str) -> int:
    return len((text or "").split())


def _mentions_university_keywords(text: str) -> bool:
    t = (text or "").lower()
    return any(
        kw in t
        for kw in [
            "university",
            "college",
            "program",
            "department",
            "msc",
            "m.s.",
            "phd",
            "mba",
            "ms ",
            "bsc",
            "ma ",
            "m tech",
        ]
    )


def _style_tone_markers_for_preset(style_preset: str) -> list[str]:
    style = (style_preset or "").strip().lower()
    if style == "friendly":
        return ["could you", "let me know", "when you get a chance", "thanks", "i really appreciate", "would you mind"]
    if style == "persuasive":
        return ["i respectfully request", "i would be grateful", "would greatly appreciate", "this would allow", "please consider"]
    if style == "creative":
        return ["if possible", "would it be possible", "quick favor", "one more thing", "i'm reaching out"]
    # professional default
    return ["kindly", "at your earliest convenience", "your consideration", "i would appreciate", "sincerely", "regards", "i look forward"]


def _word_overlap_faithfulness(source_text: str, draft_text: str) -> float:
    """
    Fallback faithfulness scorer when sentence-transformers is unavailable.
    Measures what fraction of meaningful source words appear in the draft.
    Returns 0.0–1.0.
    """
    stop_words = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "up", "out",
        "if", "or", "and", "but", "not", "no", "so", "than", "too", "very",
        "just", "that", "this", "these", "those", "what", "which", "who",
        "when", "where", "how", "all", "each", "any", "both", "more", "most",
        "other", "some", "such", "only", "own", "same", "also", "am", "an",
    }
    src_words = set(re.findall(r'\b[a-z]{3,}\b', source_text.lower())) - stop_words
    draft_words = set(re.findall(r'\b[a-z]{3,}\b', draft_text.lower())) - stop_words
    if not src_words:
        return 0.5  # neutral when source has no meaningful words
    overlap = src_words & draft_words
    return round(len(overlap) / len(src_words), 4)


def _score_intent_and_hallucination(req: DraftRequest, draft: str) -> tuple[float, float, dict]:
    combined_user = f"{req.raw_notes}\n{req.user_answers}".strip().lower()
    combined_draft = (draft or "").lower()

    purpose_l = (req.purpose or "").lower()
    wants_extension = (
        "extension" in combined_user
        or "extend" in combined_user
        or "deadline" in combined_user
        or "extension" in purpose_l
        or "deadline" in purpose_l
    )
    mentions_deposit = (
        "deposit" in combined_user
        or "tuition" in combined_user
        or "fee" in combined_user
        or "payment" in combined_user
    )
    mentions_university = (
        "university" in combined_user
        or "college" in combined_user
        or "program" in combined_user
        or "department" in combined_user
    )

    has_date_in_user = _has_date_like_text(combined_user)
    has_amount_in_user = _has_amount_like_text(combined_user)

    # Required coverage (only when user already provided the critical info type).
    required_total = 0
    hits = 0

    if wants_extension and has_date_in_user:
        required_total += 1
        if _has_date_like_text(combined_draft):
            hits += 1

    if mentions_deposit and has_amount_in_user:
        required_total += 1
        if _has_amount_like_text(combined_draft):
            hits += 1

    if mentions_university:
        required_total += 1
        if _mentions_university_keywords(combined_draft):
            hits += 1

    intent_score = (hits / required_total) if required_total else 0.0

    # Hallucination penalties: if user did NOT provide a detail, penalize if draft contains it.
    halluc_penalty = 0.0
    halluc_notes: list[str] = []

    if wants_extension and not has_date_in_user and _has_date_like_text(combined_draft):
        halluc_penalty -= 1.0
        halluc_notes.append("contains_date_without_user_date")

    if mentions_deposit and not has_amount_in_user and _has_amount_like_text(combined_draft):
        halluc_penalty -= 1.0
        halluc_notes.append("contains_amount_without_user_amount")

    # Tone score (preset marker match).
    markers = _style_tone_markers_for_preset(req.style_preset)
    marker_hits = sum(1 for m in markers if m.lower() in combined_draft)
    tone_score = 0.0
    if markers:
        tone_score = marker_hits / len(markers)

    detail = {
        "intent_required_total": required_total,
        "intent_hits": hits,
        "intent_score": round(intent_score, 3),
        "halluc_penalty": round(halluc_penalty, 3),
        "halluc_notes": halluc_notes,
        "tone_marker_hits": marker_hits,
        "tone_score": round(tone_score, 3),
    }

    return intent_score, halluc_penalty, detail


def _score_next_step_by_preset(req: DraftRequest, draft: str) -> tuple[float, dict]:
    """
    Scores next-step language based on the user's style preset.
    Higher score means the draft contains phrases that match that preset's tone.
    """
    draft_lower = (draft or "").lower()
    style = (req.style_preset or "").strip().lower()

    preset_key = "professional"
    if style == "friendly":
        preset_key = "friendly"
    elif style == "persuasive":
        preset_key = "persuasive"
    elif style == "creative":
        preset_key = "creative"

    phrase_db: dict[str, list[tuple[str, float]]] = {
        "professional": [
            ("please let me know", 1.0),
            ("at your earliest convenience", 1.0),
            ("kindly", 0.6),
            ("i would appreciate", 0.9),
            ("i look forward", 0.7),
            ("your consideration", 0.8),
            ("please review", 0.6),
            ("would you be able", 0.7),
            ("please consider", 0.7),
        ],
        "friendly": [
            ("could you", 0.7),
            ("when you get a chance", 1.0),
            ("let me know", 0.6),
            ("thanks so much", 0.6),
            ("i really appreciate", 0.7),
            ("would you mind", 0.5),
        ],
        "persuasive": [
            ("i would be grateful", 1.0),
            ("i respectfully request", 1.0),
            ("would greatly appreciate", 1.0),
            ("i hope you can", 0.7),
            ("this would allow", 0.7),
            ("therefore", 0.35),
            ("please consider", 0.8),
        ],
        "creative": [
            ("if possible", 0.8),
            ("would it be possible", 0.8),
            ("i'd love to", 0.7),
            ("quick favor", 0.6),
            ("i'm reaching out", 0.5),
            ("one more thing", 0.25),
        ],
        "default": [],
    }

    phrases = phrase_db.get(preset_key, phrase_db["professional"])
    total_weight = sum(w for _, w in phrases) or 1.0
    matched: list[tuple[str, float]] = []
    matched_weight = 0.0
    for phrase, weight in phrases:
        if phrase in draft_lower:
            matched.append((phrase, weight))
            matched_weight += float(weight)

    # Map matched_weight to (-0.5 .. +1.5)
    if matched_weight <= 0:
        delta = -0.5
    else:
        ratio = matched_weight / total_weight
        delta = -0.5 + 2.0 * ratio
        delta = max(-0.5, min(1.5, delta))

    detail = {
        "style_preset": req.style_preset,
        "preset_key": preset_key,
        "matched_phrases": [{"phrase": p, "weight": w} for p, w in matched],
        "matched_weight": round(matched_weight, 3),
        "total_weight": round(total_weight, 3),
        "delta": round(delta, 3),
    }
    return delta, detail


def _score_draft_candidate(req: DraftRequest, draft: str) -> tuple[float, dict]:
    """
    Lightweight deterministic scoring to select best draft variant.
    Higher score is better.
    """
    rubric, _ = _basic_rubric_check(draft, req)

    # --- Component contributions (weighted) ---
    closing_contrib = 1.0 if rubric.get("has_closing") else -0.5

    # Next-step language scored using preset-specific phrase mini-database.
    next_step_contrib, preset_detail = _score_next_step_by_preset(req, draft)

    short_contrib = 0.8 if rubric.get("not_too_short") else -1.0

    draft_lower = (draft or "").lower()
    has_subject = "subject:" in draft_lower
    if (req.channel or "").lower() == "email":
        # If user asked for subject, reward it; otherwise neutral.
        if req.include_subject:
            subject_contrib = 0.7 if has_subject else -0.8
        else:
            subject_contrib = 0.0
    else:
        # Non-email channels should avoid subject line.
        subject_contrib = -1.0 if has_subject else 0.4

    lo, hi = _target_word_range(req.word_size)
    wc = _count_words(draft)
    if lo <= wc <= hi:
        word_size_contrib = 1.0
    else:
        # Soft penalty based on distance.
        if wc < lo:
            word_size_contrib = -min(1.5, (lo - wc) / max(lo, 1))
        else:
            word_size_contrib = -min(1.5, (wc - hi) / max(hi, 1))

    intent_score, halluc_penalty, intent_detail = _score_intent_and_hallucination(req, draft)
    # Convert intent/hallucination into contributions that align with the rest of the scale.
    intent_contrib = intent_score  # 0..1
    hallucination_contrib = halluc_penalty  # negative or 0

    # Semantic faithfulness scoring.
    # Priority: sentence-transformers → word-overlap fallback.
    faithfulness_score = 0.0
    source_text = f"{req.raw_notes}\n{req.user_answers}".strip()
    try:
        from src.scoring.faithfulness import get_faithfulness_scorer
        if source_text and draft:
            faithfulness_score = get_faithfulness_scorer().score(source_text, draft)
    except Exception:
        # Fallback: simple word-overlap ratio (better than returning 0).
        if source_text and draft:
            faithfulness_score = _word_overlap_faithfulness(source_text, draft)

    markers = _style_tone_markers_for_preset(req.style_preset)
    tone_markers_found = [m for m in markers if m.lower() in draft_lower]
    tone_markers_missing = [m for m in markers if m.lower() not in draft_lower]
    tone_marker_hits = len(tone_markers_found)
    tone_score = (tone_marker_hits / len(markers)) if markers else 0.0

    score = (
        SCORING_WEIGHTS["w_closing"] * closing_contrib
        + SCORING_WEIGHTS["w_next_step"] * next_step_contrib
        + SCORING_WEIGHTS["w_short"] * short_contrib
        + SCORING_WEIGHTS["w_subject"] * subject_contrib
        + SCORING_WEIGHTS["w_word_size"] * word_size_contrib
        + SCORING_WEIGHTS["w_intent"] * intent_contrib
        + SCORING_WEIGHTS["w_hallucination"] * hallucination_contrib
        + SCORING_WEIGHTS["w_tone"] * tone_score
        + SCORING_WEIGHTS.get("w_faithfulness", 0.8) * faithfulness_score
    )

    # Build faithfulness explanation for the UI.
    faithfulness_explanation: Dict[str, Any] = {"score": round(faithfulness_score, 4), "method": "word_overlap"}
    try:
        from src.scoring.faithfulness import get_faithfulness_scorer
        _ = get_faithfulness_scorer()
        faithfulness_explanation["method"] = "sentence_embeddings"
    except Exception:
        pass
    if source_text and draft:
        # Provide word-level detail for explanation regardless of method.
        stop_words = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about", "up", "out",
            "if", "or", "and", "but", "not", "no", "so", "than", "too", "very",
            "just", "that", "this", "these", "those", "what", "which", "who",
            "when", "where", "how", "all", "each", "any", "both", "more", "most",
            "other", "some", "such", "only", "own", "same", "also", "am", "an",
        }
        src_words = set(re.findall(r'\b[a-z]{3,}\b', source_text.lower())) - stop_words
        draft_words = set(re.findall(r'\b[a-z]{3,}\b', draft_lower)) - stop_words
        found_words = sorted(src_words & draft_words)
        missing_words = sorted(src_words - draft_words)
        faithfulness_explanation["source_keywords"] = list(src_words)[:20]
        faithfulness_explanation["found_in_draft"] = found_words[:15]
        faithfulness_explanation["missing_from_draft"] = missing_words[:15]

    details: Dict[str, Any] = {
        "word_count": wc,
        "has_subject": has_subject,
        "next_step_preset_detail": preset_detail,
        "rubric": rubric,
        "intent_detail": intent_detail,
        "faithfulness_score": round(faithfulness_score, 4),
        "faithfulness_explanation": faithfulness_explanation,
        "tone_marker_hits": tone_marker_hits,
        "tone_score": round(tone_score, 3),
        "tone_markers_found": tone_markers_found,
        "tone_markers_missing": tone_markers_missing,
        "style_preset": req.style_preset,
        "component": {
            "closing_contrib": closing_contrib,
            "next_step_contrib": next_step_contrib,
            "short_contrib": short_contrib,
            "subject_contrib": subject_contrib,
            "word_size_contrib": word_size_contrib,
            "intent_contrib": round(intent_contrib, 3),
            "hallucination_contrib": hallucination_contrib,
            "faithfulness_contrib": round(faithfulness_score, 4),
            "tone_contrib": round(tone_score, 3),
        },
        "score": round(score, 3),
    }
    return score, details


def _generate_draft_variants(
    req: DraftRequest,
    llm_client: Any,
    draft_prompt: str,
    options: dict,
) -> tuple[str, dict]:
    """
    Generates multiple draft variants and returns best draft + metadata.
    """
    variants = max(1, min(5, int(getattr(req, "draft_variants", 1) or 1)))
    candidates: list[dict] = []

    for idx in range(variants):
        draft_options = dict(options)
        if "seed" in draft_options:
            # Force distinct generations per candidate while preserving reproducibility.
            draft_options["seed"] = int(draft_options["seed"]) + idx

        if hasattr(llm_client, "generate_with_options"):
            draft_text = llm_client.generate_with_options(draft_prompt, options=draft_options)
        else:
            draft_text = llm_client.generate(draft_prompt)

        score, details = _score_draft_candidate(req, draft_text)
        next_step = details.get("next_step_preset_detail", {}) if isinstance(details, dict) else {}
        candidates.append(
            {
                "index": idx,
                "score": round(score, 3),
                "word_count": details["word_count"],
                "has_subject": details["has_subject"],
                "next_step_delta": next_step.get("delta"),
                "next_step_matched_weight": next_step.get("matched_weight"),
                "next_step_matched_phrases": next_step.get("matched_phrases", []),
                "intent_score": details.get("intent_detail", {}).get("intent_score") if isinstance(details.get("intent_detail"), dict) else None,
                "faithfulness_score": details.get("faithfulness_score"),
                "faithfulness_explanation": details.get("faithfulness_explanation", {}),
                "hallucination_contrib": details.get("intent_detail", {}).get("halluc_penalty") if isinstance(details.get("intent_detail"), dict) else None,
                "halluc_notes": details.get("intent_detail", {}).get("halluc_notes", []) if isinstance(details.get("intent_detail"), dict) else [],
                "tone_score": details.get("tone_score"),
                "tone_marker_hits": details.get("tone_marker_hits"),
                "tone_markers_found": details.get("tone_markers_found", []),
                "tone_markers_missing": details.get("tone_markers_missing", []),
                "style_preset": details.get("style_preset", ""),
                "component": details.get("component", {}),
                "text": draft_text,
            }
        )

    # Heuristic best (fallback).
    heuristic_best = sorted(candidates, key=lambda c: (-c["score"], c["index"]))[0]

    # Optional LLM judge best.
    judge_best_index, judge_meta = _judge_draft_candidates(req, candidates, llm_client=llm_client, options=options)
    if judge_best_index is not None:
        matched = [c for c in candidates if c.get("index") == judge_best_index]
        if matched:
            best = matched[0]
        else:
            best = heuristic_best
    else:
        best = heuristic_best

    metadata = {
        "draft_variants_requested": variants,
        "draft_variants_scored": [
            {k: v for k, v in c.items() if k != "text"} for c in candidates
        ],
        "heuristic_best_index": heuristic_best["index"],
        "heuristic_best_score": heuristic_best["score"],
        "judge_best_index": judge_best_index,
        "judge_meta": judge_meta,
        "selected_variant_index": best["index"],
        "selected_variant_score": best["score"],
    }
    return best["text"], metadata


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort extraction of a single JSON object from model output.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None


def _judge_draft_candidates(
    req: DraftRequest,
    candidates: List[dict],
    *,
    llm_client: Any,
    options: dict,
) -> tuple[Optional[int], dict]:
    """
    Optional LLM judge to choose the best draft among candidates.
    Enabled via env var `AGENT_JUDGE_ENABLED=1`.
    Returns: (best_index_or_none, judge_metadata)
    """
    if (os.getenv("AGENT_JUDGE_ENABLED", "0") or "").strip() not in {"1", "true", "True", "yes"}:
        return None, {"judge_enabled": False}

    # Truncate candidate text to keep judge prompt size reasonable.
    candidate_lines: list[str] = []
    for c in candidates:
        text = c.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text_preview = text[:900]
        candidate_lines.append(
            f"Index {c.get('index')}\n{text_preview}"
        )

    candidates_block = "\n\n---\n\n".join(candidate_lines)

    judge_prompt = JUDGE_DRAFT_PROMPT_TEMPLATE.format(
        channel=req.channel,
        purpose=req.purpose,
        audience=req.audience,
        tone=req.tone,
        style_preset=req.style_preset,
        word_size=req.word_size,
        include_subject=str(req.include_subject).lower(),
        finalize_requested=str(req.finalize_requested).lower(),
        raw_notes=req.raw_notes,
        user_answers=req.user_answers,
        candidates=candidates_block,
    )

    judge_options = dict(options)
    # Keep judge consistent.
    judge_options["temperature"] = min(0.4, float(judge_options.get("temperature", 0.2)))
    judge_options["top_p"] = min(0.9, float(judge_options.get("top_p", 0.9)))

    if hasattr(llm_client, "generate_with_options"):
        raw = llm_client.generate_with_options(judge_prompt, options=judge_options)
    else:
        raw = llm_client.generate(judge_prompt)

    parsed = _extract_json_object(raw)
    if not parsed or not isinstance(parsed, dict):
        return None, {"judge_enabled": True, "parse_ok": False, "raw": raw[:500]}

    best_index = parsed.get("best_index")
    try:
        best_index_int = int(best_index)
    except Exception:
        best_index_int = None

    meta = {
        "judge_enabled": True,
        "parse_ok": True,
        "best_index": best_index_int,
        "judge_payload_preview": raw[:500],
        "judged_candidates": parsed.get("candidates", []),
    }
    return best_index_int, meta


def _has_date_like_text(text: str) -> bool:
    t = (text or "").lower()
    # Common date patterns: 12/03, 12-03-2026, "March 12", weekday names, times like 5pm, 5:30 pm
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b",
        r"\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    ]
    return any(re.search(p, t) for p in date_patterns)


def _has_amount_like_text(text: str) -> bool:
    t = (text or "").lower()
    amount_patterns = [
        r"(usd|inr|eur|gbp|rs\.?|dollars?|bucks|pounds?)\s*\d+(\.\d+)?",
        r"(\$|€|£|₹)\s*\d+(\.\d+)?",
        r"\b(amount|deposit|tuition|fee|payment)\b.{0,24}\b\d+(\.\d+)?\b",
    ]
    return any(re.search(p, t) for p in amount_patterns)


def _build_critical_questions(req: DraftRequest) -> list[str]:
    """
    Heuristic critical questions to ensure correctness.
    This is intentionally conservative: it errs on asking for missing key details.
    """
    combined = f"{req.raw_notes}\n{req.user_answers}".strip().lower()
    purpose_l = (req.purpose or "").lower()

    wants_extension = ("extension" in combined) or ("extend" in combined) or ("deadline" in combined) or ("extension" in purpose_l) or ("deadline" in purpose_l)
    mentions_deposit = ("deposit" in combined) or ("tuition" in combined) or ("fee" in combined) or ("payment" in combined)
    mentions_university = ("university" in combined) or ("college" in combined) or ("program" in combined) or ("department" in combined)

    critical: list[str] = []
    if wants_extension and not _has_date_like_text(combined):
        critical.append("What is the exact new deadline/date (including time if relevant) you want to request?")
    if mentions_deposit and not _has_amount_like_text(combined):
        critical.append("What is the deposit/fee amount and currency involved?")
    if mentions_university:
        # We can't reliably extract names, but we can ask the user for it when the topic is present.
        critical.append("Which university/program is this for (include department if you know it)?")

    # If the user mentions an extension but we still have no critical items,
    # ask about the requested action/outcome.
    if wants_extension and not critical:
        critical.append("What exactly are you requesting (approve/reschedule/confirm)?")

    # If still empty, ask who it’s for (helps correctness across channels).
    if not critical:
        critical.append("Who is the recipient (name/title), and what outcome do you want?")

    return critical[:6]


def _build_optional_questions(req: DraftRequest) -> list[str]:
    channel_l = (req.channel or "").lower()
    purpose_l = (req.purpose or "").lower()
    optional: list[str] = []

    if channel_l == "email":
        optional.append("What greeting and sign-off style do you prefer (formal vs friendly)?")
        optional.append("Do you want to include a short reason/apology, or keep it strictly formal?")
    elif "whatsapp" in channel_l:
        optional.append("Should it be very short (1-2 sentences) or medium (a few sentences)?")
        optional.append("Do you want it to sound more casual or more formal?")
    else:
        optional.append("Should the message be brief (Teams-style) or slightly detailed?")
        optional.append("Do you want to propose a next step/time for them to respond?")

    if "extension" in purpose_l or "deadline" in purpose_l:
        optional.append("Do you want to include a reason for the extension (if yes, what is it)?")

    return optional[:6]


def _deduplicate_questions(questions: list[str]) -> list[str]:
    """
    Remove near-duplicate questions.
    Uses sentence-transformers cosine similarity if available, otherwise substring matching.
    """
    if not questions:
        return []

    try:
        from src.scoring.faithfulness import get_faithfulness_scorer
        import numpy as np

        scorer = get_faithfulness_scorer()
        scorer._load()
        embs = scorer._model.encode(questions, convert_to_numpy=True)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = norms @ norms.T

        keep = []
        for i in range(len(questions)):
            is_dup = False
            for j in keep:
                if sim_matrix[i][j] > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)
        return [questions[i] for i in keep]
    except Exception:
        pass

    # Fallback: simple substring containment check
    unique: list[str] = []
    for q in questions:
        q_lower = q.lower().strip()
        if not any(q_lower in existing.lower() or existing.lower() in q_lower for existing in unique):
            unique.append(q)
    return unique


def _clarify_first(req: DraftRequest, *, llm_client: Any, options: dict) -> tuple[bool, list[str]]:
    """
    Returns: (proceed, questions)
    Ensembles heuristic questions with LLM clarification (if enabled).
    """
    user_answers_empty = not (req.user_answers or "").strip()

    # Heuristic critical questions (guarantees correctness even if the model is overconfident).
    critical_questions = _build_critical_questions(req)
    optional_questions = _build_optional_questions(req)

    # LLM clarification (opt-in via env var, default enabled).
    llm_proceed = True
    llm_questions: list[str] = []
    if os.getenv("LLM_CLARIFY_ENABLED", "1").strip() in {"1", "true", "True", "yes"}:
        try:
            clarify_prompt = CLARIFY_PROMPT_TEMPLATE.format(
                channel=req.channel,
                purpose=req.purpose,
                audience=req.audience,
                tone=req.tone,
                raw_notes=req.raw_notes,
                user_answers=req.user_answers or "",
            )
            clarify_options = dict(options)
            clarify_options["temperature"] = 0.3
            if hasattr(llm_client, "generate_with_options"):
                raw = llm_client.generate_with_options(clarify_prompt, options=clarify_options)
            else:
                raw = llm_client.generate(clarify_prompt)
            parsed = _extract_json_object(raw)
            if parsed and isinstance(parsed, dict):
                llm_proceed = bool(parsed.get("proceed", True))
                llm_questions = parsed.get("questions", [])
                if not isinstance(llm_questions, list):
                    llm_questions = []
        except Exception:
            pass  # fall back to heuristic-only

    # Ensemble: critical heuristic first, then LLM, then optional heuristic.
    # Deduplicate and cap at 6.
    combined_questions = _deduplicate_questions(
        critical_questions + llm_questions + optional_questions
    )[:6]

    # If user answers are empty, always ask questions first.
    if user_answers_empty:
        return False, combined_questions

    # If user provided answers, check if critical info is still missing.
    heuristic_proceed = True
    combined = f"{req.raw_notes}\n{req.user_answers}".strip().lower()
    if ("extension" in combined or "extend" in combined or "deadline" in combined) and not _has_date_like_text(combined):
        heuristic_proceed = False
    if (("deposit" in combined) or ("tuition" in combined) or ("fee" in combined) or ("payment" in combined)) and not _has_amount_like_text(combined):
        heuristic_proceed = False

    # Conservative: proceed only if BOTH heuristic and LLM agree.
    proceed = heuristic_proceed and llm_proceed

    if not proceed:
        return False, combined_questions

    return True, []


def run_draft_to_ready(req: DraftRequest, *, llm_client: Any) -> DraftResponse:
    """
    Agent workflow:
    1) Draft generation from user notes
    2) Rubric check (lightweight)
    3) Optional finalize/edit generation
    """
    options = _ollama_options_from_req(req)
    finalize_options = dict(options)
    if "seed" in finalize_options:
        # Use a slightly different seed for the finalize/edit step
        # so it does not become overly similar to the first draft.
        finalize_options["seed"] = int(finalize_options["seed"]) + 1
    word_count_block = _word_count_block(req)

    # Step 0: clarity-first (ask questions; but always draft so the user sees progress)
    proceed, clarify_questions = _clarify_first(req, llm_client=llm_client, options=options)

    combined_raw_notes = req.raw_notes
    if req.user_answers and req.user_answers.strip():
        combined_raw_notes = f"{req.raw_notes.strip()}\n\nUser answers:\n{req.user_answers.strip()}"

    # Step 1: draft (best-of-N selection)
    draft_prompt = DRAFT_PROMPT_TEMPLATE.format(
        purpose=req.purpose,
        tone=req.tone,
        audience=req.audience,
        channel=req.channel,
        include_subject=str(req.include_subject).lower(),
        word_count_block=word_count_block,
        raw_notes=combined_raw_notes,
    )
    draft, variant_meta = _generate_draft_variants(
        req=req,
        llm_client=llm_client,
        draft_prompt=draft_prompt,
        options=options,
    )

    # Step 2: self-check (used for rubric metadata only; questions come from clarification step)
    rubric, _rubric_questions = _basic_rubric_check(draft, req)
    rubric.update(variant_meta)

    # Step 2b: NLI hallucination detection (optional, graceful fallback).
    try:
        from src.scoring.hallucination import get_hallucination_detector
        source_text = f"{req.raw_notes}\n{req.user_answers}".strip()
        if source_text and draft:
            halluc_result = get_hallucination_detector().detect(source_text, draft)
            rubric["hallucination_score"] = halluc_result.get("hallucination_score", 0.0)
            rubric["flagged_sentences"] = halluc_result.get("flagged_sentences", [])
            threshold = float(os.getenv("HALLUCINATION_THRESHOLD", "0.3"))
            if halluc_result.get("hallucination_score", 0.0) > threshold:
                rubric["hallucination_warning"] = True
    except Exception:
        pass  # sentence-transformers/cross-encoder not installed; skip

    # Step 2c: Add faithfulness score to rubric from variant metadata.
    # (Faithfulness is computed per-variant in _score_draft_candidate; surface the selected variant's score.)
    scored_variants = variant_meta.get("draft_variants_scored", [])
    selected_idx = variant_meta.get("selected_variant_index")
    for v in scored_variants:
        if v.get("index") == selected_idx and "faithfulness_score" in v:
            rubric["faithfulness_score"] = v["faithfulness_score"]
            break

    # If we still need clarification, return the draft and do NOT finalize yet.
    if not proceed:
        rubric["finalized_with"] = "skipped (needs clarification)"
        return DraftResponse(
            questions=clarify_questions[:6],
            draft=draft,
            rubric_check={"status": "needs_clarification", **rubric},
            final="",
        )

    # If user only asked for a draft, do not run finalize/edit yet.
    if not getattr(req, "finalize_requested", False):
        rubric["finalized_with"] = "skipped (draft only)"
        return DraftResponse(
            questions=[],
            draft=draft,
            rubric_check=rubric,
            final="",
        )

    # Step 3: finalize/edit (skip for mock to keep deterministic)
    if isinstance(llm_client, MockLLMClient):
        final = draft
        rubric["finalized_with"] = "mock (no edit pass)"
    else:
        rubric_notes = "; ".join([f"{k}={v}" for k, v in rubric.items()])
        finalize_prompt = FINALIZE_PROMPT_TEMPLATE.format(
            rubric_notes=rubric_notes,
            draft=draft,
            word_count_block=word_count_block,
        )
        if hasattr(llm_client, "generate_with_options"):
            final = llm_client.generate_with_options(finalize_prompt, options=finalize_options)
        else:
            final = llm_client.generate(finalize_prompt)
        rubric["finalized_with"] = "ollama finalize pass"

    return DraftResponse(
        questions=[],
        draft=draft,
        rubric_check=rubric,
        final=final,
    )

