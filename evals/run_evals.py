import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.workflow import run_draft_to_ready  # noqa: E402
from src.llm.mock_client import MockLLMClient  # noqa: E402
from src.llm.ollama_client import get_ollama_client  # noqa: E402
from src.schemas.models import DraftRequest  # noqa: E402


WORD_SIZE_RANGES = {
    # Soft ranges used for generic quality checks.
    # Kept broad because model outputs vary (especially across local models).
    "Small": (20, 160),
    "Medium": (20, 300),
    "Large": (30, 420),
}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text: str) -> int:
    return len((text or "").split())


def _new_category_stats() -> Dict[str, Dict[str, int]]:
    return {
        "clarification": {"passed": 0, "total": 0},
        "content_presence": {"passed": 0, "total": 0},
        "channel_rules": {"passed": 0, "total": 0},
        "size_range": {"passed": 0, "total": 0},
    }


def _record(category_stats: Dict[str, Dict[str, int]], category: str, ok: bool) -> None:
    category_stats[category]["total"] += 1
    if ok:
        category_stats[category]["passed"] += 1


def _merge_category_stats(
    agg: Dict[str, Dict[str, int]], per_case: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    for category, vals in per_case.items():
        agg[category]["passed"] += vals["passed"]
        agg[category]["total"] += vals["total"]
    return agg


def check_case(
    case: Dict[str, Any], response: Any
) -> Tuple[bool, List[str], Dict[str, Dict[str, int]], Dict[str, Any]]:
    issues: List[str] = []
    expect = case.get("expect", {})
    category_stats = _new_category_stats()
    selector_meta: Dict[str, Any] = {}

    needs_clarification = expect.get("needs_clarification")
    if needs_clarification is True:
        status = (response.rubric_check or {}).get("status")
        ok = status == "needs_clarification"
        _record(category_stats, "clarification", ok)
        if not ok:
            issues.append("Expected needs_clarification status")

    has_draft = expect.get("has_draft")
    if has_draft is True:
        ok = bool(response.draft)
        _record(category_stats, "content_presence", ok)
        if not ok:
            issues.append("Expected non-empty draft")
    if has_draft is False:
        ok = not bool(response.draft)
        _record(category_stats, "content_presence", ok)
        if not ok:
            issues.append("Expected empty draft")

    has_final = expect.get("has_final")
    if has_final is True:
        ok = bool(response.final)
        _record(category_stats, "content_presence", ok)
        if not ok:
            issues.append("Expected non-empty final")
    if has_final is False:
        ok = not bool(response.final)
        _record(category_stats, "content_presence", ok)
        if not ok:
            issues.append("Expected empty final")

    min_questions = expect.get("min_questions")
    if isinstance(min_questions, int):
        ok = len(response.questions or []) >= min_questions
        _record(category_stats, "content_presence", ok)
        if not ok:
            issues.append(f"Expected at least {min_questions} questions")

    text_for_checks = response.final or response.draft or ""
    text_lower = text_for_checks.lower()

    if expect.get("forbid_subject") is True:
        ok = "subject:" not in text_lower
        _record(category_stats, "channel_rules", ok)
        if not ok:
            issues.append("Subject line must not appear for this case")

    if expect.get("contains_subject_when_email") is True:
        ok = "subject:" in text_lower
        _record(category_stats, "channel_rules", ok)
        if not ok:
            issues.append("Expected subject line for email case")

    # Soft size check only when we have output text.
    input_data = case.get("input", {})
    if text_for_checks:
        size = input_data.get("word_size", "Medium")
        lo, hi = WORD_SIZE_RANGES.get(size, WORD_SIZE_RANGES["Medium"])
        wc = word_count(text_for_checks)
        ok = lo <= wc <= hi
        _record(category_stats, "size_range", ok)
        if not ok:
            issues.append(f"Word size mismatch: got {wc}, expected approx {lo}-{hi}")

    # --- Selector evaluation (best-of-N) ---
    rubric_check = response.rubric_check or {}
    scored = rubric_check.get("draft_variants_scored") or []
    selected_idx = rubric_check.get("selected_variant_index")
    selected_score = rubric_check.get("selected_variant_score")

    channel_expect_forbid_subject = expect.get("forbid_subject") is True
    channel_expect_contains_subject = expect.get("contains_subject_when_email") is True

    size_target = input_data.get("word_size", "Medium")
    lo, hi = WORD_SIZE_RANGES.get(size_target, WORD_SIZE_RANGES["Medium"])

    passing_scores: list[float] = []
    selected_pass = False
    max_passing_score = None

    for cand in scored:
        cand_idx = cand.get("index")
        cand_has_subject = cand.get("has_subject")
        cand_wc = cand.get("word_count")
        cand_score = cand.get("score")

        channel_ok = True
        if channel_expect_forbid_subject:
            channel_ok = cand_has_subject is False
        if channel_expect_contains_subject:
            channel_ok = cand_has_subject is True

        size_ok = True
        if cand_wc is not None:
            size_ok = lo <= int(cand_wc) <= hi

        cand_pass = bool(channel_ok) and bool(size_ok)

        if cand_pass and cand_score is not None:
            passing_scores.append(float(cand_score))

        if cand_idx == selected_idx:
            selected_pass = cand_pass

    if passing_scores:
        max_passing_score = max(passing_scores)
    if max_passing_score is not None and selected_score is not None:
        selector_meta["selection_is_best_passing"] = float(selected_score) >= float(max_passing_score) - 1e-9
    else:
        selector_meta["selection_is_best_passing"] = None
    selector_meta["selected_variant_channel_and_size_ok"] = selected_pass
    selector_meta["selector_max_passing_score"] = max_passing_score
    selector_meta["selected_variant_score_for_selector"] = selected_score

    return len(issues) == 0, issues, category_stats, selector_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation harness for Draft-to-Ready agent.")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "evals" / "cases.json"),
        help="Path to eval cases JSON file",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama model instead of mock client",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly use MockLLMClient (default behavior, useful for CI clarity)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0"),
        help="Ollama model tag when --use-ollama is enabled",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    cases = load_cases(cases_path)

    if args.use_ollama:
        llm_client = get_ollama_client(model_name=args.model)
        client_name = f"ollama:{args.model}"
    else:
        llm_client = MockLLMClient()
        client_name = "mock"

    results: List[Dict[str, Any]] = []
    passed = 0
    category_breakdown = _new_category_stats()
    selection_best_passing_accum: List[bool] = []

    for case in cases:
        req = DraftRequest(**case["input"])
        resp = run_draft_to_ready(req, llm_client=llm_client)
        ok, issues, case_category_stats, selector_meta = check_case(case, resp)
        category_breakdown = _merge_category_stats(category_breakdown, case_category_stats)
        if ok:
            passed += 1
        results.append(
            {
                "id": case.get("id"),
                "pass": ok,
                "issues": issues,
                "questions_count": len(resp.questions or []),
                "has_draft": bool(resp.draft),
                "has_final": bool(resp.final),
                "rubric_status": (resp.rubric_check or {}).get("status", "ok"),
                "category_scores": case_category_stats,
                "selector_meta": selector_meta,
            }
        )
        sel_best = selector_meta.get("selection_is_best_passing")
        if isinstance(sel_best, (int, float)):
            selection_best_passing_accum.append(bool(sel_best))

    total = len(results)
    print(f"\nEvaluation client: {client_name}")
    print(f"Passed: {passed}/{total} ({(100.0 * passed / max(total, 1)):.1f}%)\n")

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"  - {issue}")

    print("\nCategory-wise score breakdown:")
    for category, vals in category_breakdown.items():
        total_checks = vals["total"]
        passed_checks = vals["passed"]
        pct = 100.0 * passed_checks / max(total_checks, 1)
        print(f"- {category}: {passed_checks}/{total_checks} ({pct:.1f}%)")

    if selection_best_passing_accum:
        sel_acc = 100.0 * sum(1 for x in selection_best_passing_accum if x) / len(selection_best_passing_accum)
        print(f"Selector best-passing accuracy: {sel_acc:.1f}% ({len(selection_best_passing_accum)} cases evaluated)")

    out_path = ROOT / "evals" / "last_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "client": client_name,
                    "passed_cases": passed,
                    "total_cases": total,
                    "pass_rate": round(100.0 * passed / max(total, 1), 2),
                    "category_breakdown": category_breakdown,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved detailed results: {out_path}")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

