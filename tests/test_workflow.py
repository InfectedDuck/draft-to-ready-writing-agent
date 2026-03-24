from src.agent.workflow import (
    run_draft_to_ready,
    _clarify_first,
    _deduplicate_questions,
    _ollama_options_from_req,
)
from src.llm.mock_client import MockLLMClient
from src.schemas.models import DraftRequest


def test_mock_workflow_produces_final():
    req = DraftRequest(
        raw_notes="I need to request an extension because I was sick. Deadline was last Friday.",
        purpose="email reply",
        audience="teacher",
        tone="formal",
        include_subject=True,
    )
    resp = run_draft_to_ready(req, llm_client=MockLLMClient())
    # Multi-round: without user answers, we should get questions and a draft (but no final).
    assert resp.questions
    assert resp.draft
    assert resp.final == ""
    assert isinstance(resp.rubric_check, dict)


def test_mock_workflow_after_answers_drafts():
    req = DraftRequest(
        raw_notes="I need to request an extension because I was sick.",
        user_answers="The deadline I'm requesting is Monday, 5pm. Please approve the extension.",
        purpose="email reply",
        audience="teacher",
        tone="formal",
        include_subject=True,
        finalize_requested=True,
    )
    resp = run_draft_to_ready(req, llm_client=MockLLMClient())
    assert resp.draft
    assert resp.final


def test_ensemble_deduplicates_questions():
    """Verify the clarification ensemble produces deduplicated questions."""
    req = DraftRequest(
        raw_notes="I need to request an extension for my assignment. The university deadline is approaching.",
        user_answers="",
        purpose="extension request",
        audience="professor",
        tone="formal",
    )
    options = _ollama_options_from_req(req)
    proceed, questions = _clarify_first(req, llm_client=MockLLMClient(), options=options)

    assert not proceed, "Should not proceed without user answers"
    assert len(questions) <= 6, "Questions should be capped at 6"
    assert len(questions) > 0, "Should have at least 1 question"

    # Check no exact duplicates
    assert len(questions) == len(set(questions)), "Should have no exact duplicates"

    # Check no near-duplicates (simple substring check)
    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions):
            if i != j:
                assert q1.lower() not in q2.lower(), (
                    f"Question '{q1}' is a substring of '{q2}'"
                )


def test_faithfulness_scorer_structure():
    """Test that FaithfulnessScorer has the expected interface."""
    from src.scoring.faithfulness import FaithfulnessScorer, get_faithfulness_scorer

    scorer = get_faithfulness_scorer()
    assert isinstance(scorer, FaithfulnessScorer)
    assert hasattr(scorer, "score")
    assert hasattr(scorer, "detail")
    # We don't call score/detail here since sentence-transformers may not be installed.


def test_hallucination_detector_structure():
    """Test that HallucinationDetector has the expected interface."""
    from src.scoring.hallucination import HallucinationDetector, get_hallucination_detector

    detector = get_hallucination_detector()
    assert isinstance(detector, HallucinationDetector)
    assert hasattr(detector, "detect")
