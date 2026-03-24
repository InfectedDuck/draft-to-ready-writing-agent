# Implementation Prompt for Draft-to-Ready Writing Agent Upgrades

## Project Context

This is a "Draft-to-Ready Writing Agent" — a Gradio-based app that takes raw notes/bullets from a user and generates polished, ready-to-send messages (emails, WhatsApp, Teams). The architecture:

```
app.py                          # Gradio 6 UI (838 lines)
src/
  schemas/models.py             # DraftRequest, DraftResponse (Pydantic)
  agent/workflow.py             # Core agent: clarification gate → multi-variant draft → heuristic scoring → optional LLM judge → rubric → finalize
  prompts/templates.py          # DRAFT_PROMPT_TEMPLATE, FINALIZE_PROMPT_TEMPLATE, CLARIFY_PROMPT_TEMPLATE (unused), JUDGE_DRAFT_PROMPT_TEMPLATE
  llm/ollama_client.py          # OllamaClient (HTTP to Ollama /api/generate)
  llm/mock_client.py            # MockLLMClient (deterministic stub for tests/fallback)
evals/
  run_evals.py                  # Batch eval harness: runs cases, checks expectations, writes last_results.json
  calibrate_scoring_weights.py  # Grid search tuning w_next_step / w_intent
  cases.json, cases_hard.json   # Eval case definitions
  scoring_weights.json          # Calibrated weights
  last_results.json             # Latest eval output
tests/
  test_workflow.py              # 2 pytest tests (mock workflow paths)
requirements.txt                # gradio, pydantic, requests, python-dotenv, pytest
```

### Current Scoring (what needs replacing/augmenting)

In `workflow.py`, `_score_intent_and_hallucination(draft, raw_notes, user_answers)` does **naive keyword matching**: it tokenizes notes into words, checks how many appear in the draft, and penalizes unknown words. `_score_draft_candidate` combines this with other heuristic checks (closing phrases, length, tone markers, subject line). Weights are loaded from `scoring_weights.json`.

### Current Clarification (what's half-built)

`CLARIFY_PROMPT_TEMPLATE` exists in `templates.py` with a JSON schema (`{proceed: bool, questions: [...]}`) but is **never called** in `workflow.py`. Clarification is 100% heuristic via `_clarify_first` → `_build_critical_questions` + `_build_optional_questions` (pattern-matching for dates, amounts, recipient names, etc.).

### Current Eval Output

`run_evals.py` prints category-level pass rates and selector stats to stdout, writes full results to `last_results.json`. No visualization.

---

## Task: Implement These 5 Upgrades

### Upgrade 1: BERTScore / Embedding-Based Faithfulness Scoring

**Goal:** Replace the naive keyword overlap in `_score_intent_and_hallucination` with semantic similarity using sentence-transformers.

**Requirements:**

1. Create `src/scoring/faithfulness.py` with:
   - A `FaithfulnessScorer` class that lazy-loads `sentence-transformers/all-MiniLM-L6-v2` on first call (so import cost is paid once).
   - Method `score(source_text: str, draft_text: str) -> float` that:
     - Splits both texts into sentences.
     - Computes embeddings for all sentences.
     - For each source sentence, finds the max cosine similarity with any draft sentence.
     - Returns the mean of these max similarities as the "faithfulness" score (0.0–1.0).
   - Method `detail(source_text: str, draft_text: str) -> dict` that returns per-source-sentence scores for debugging/dashboard use.
   - A module-level singleton `get_faithfulness_scorer()` that returns a cached instance.

2. In `workflow.py`:
   - Import and use `FaithfulnessScorer` inside `_score_intent_and_hallucination` (or a new function that replaces it).
   - The old keyword score should become a **fallback** if sentence-transformers is not installed (try/except import).
   - Integrate the faithfulness score into `_score_draft_candidate` as a weighted component (add `w_faithfulness` to `SCORING_WEIGHTS` defaults).
   - Add faithfulness score to the rubric dict so the UI can display it.

3. Add `sentence-transformers` to `requirements.txt`.

**Constraints:**
- The scorer must work offline (model downloaded once, then cached locally).
- Keep the fallback to keyword matching if the package isn't installed, so tests still pass without heavy dependencies.
- ~50-80 lines for the scorer module itself.

---

### Upgrade 2: NLI-Based Hallucination Detection

**Goal:** Detect fabricated/unsupported sentences in the draft using Natural Language Inference.

**Requirements:**

1. Create `src/scoring/hallucination.py` with:
   - A `HallucinationDetector` class that lazy-loads `cross-encoder/nli-deberta-v3-xsmall` via the `cross-encoder` or `sentence-transformers` CrossEncoder API.
   - Method `detect(source_text: str, draft_text: str, threshold: float = 0.7) -> dict` that:
     - Splits the draft into sentences.
     - For each draft sentence, runs NLI with premise=source_text, hypothesis=draft_sentence.
     - Labels each sentence as "entailed", "neutral", or "contradiction" based on the model's output.
     - Returns `{"flagged_sentences": [...], "hallucination_score": float, "details": [...]}` where `hallucination_score` is the fraction of sentences flagged (contradiction or high-confidence neutral).
   - A module-level singleton `get_hallucination_detector()`.

2. In `workflow.py`:
   - Call `HallucinationDetector.detect()` on the selected draft candidate.
   - Add hallucination results to `rubric_check` (flagged sentence count, hallucination score).
   - If hallucination_score exceeds a threshold (configurable via env var `HALLUCINATION_THRESHOLD`, default 0.3), add a warning to the rubric.
   - Like the faithfulness scorer, fall back gracefully if the model isn't available.

3. In `app.py`:
   - In `_render_rubric_html`, render hallucination flags distinctly (e.g., red warning badges for flagged sentences).

4. Add `cross-encoder` (or the appropriate package) to `requirements.txt` if not already covered by sentence-transformers.

**Constraints:**
- Must not block the UI for more than a few seconds. If the model is slow, add a note in code about optional async or caching.
- ~50-70 lines for the detector module.
- Graceful fallback if models not installed.

---

### Upgrade 3: Wire LLM Clarification + Ensemble with Heuristics

**Goal:** Actually use `CLARIFY_PROMPT_TEMPLATE` from `templates.py` and ensemble its output with the existing heuristic questions.

**Requirements:**

1. In `workflow.py`, modify `_clarify_first`:
   - After building heuristic questions (`_build_critical_questions` + `_build_optional_questions`), also call the LLM with `CLARIFY_PROMPT_TEMPLATE` (filled with the user's raw notes, purpose, audience, tone, channel).
   - Parse the LLM's JSON response (`{proceed: bool, questions: [...]}`).
   - **Ensemble logic:**
     - Union the heuristic questions and LLM questions.
     - Deduplicate by semantic similarity (if sentence-transformers is available) or by simple substring/fuzzy matching.
     - Cap at 6 total questions (prioritize critical heuristic questions first, then LLM questions, then optional heuristic questions).
     - The `proceed` decision: proceed only if BOTH heuristic and LLM agree to proceed (conservative approach).
   - If the LLM call fails (timeout, parse error), fall back silently to heuristic-only (current behavior).

2. Add a `generate_with_options` call path for the clarification LLM call, using lower temperature (0.3) for more deterministic question generation.

3. Update `MockLLMClient` to handle clarification prompts properly (it should return a JSON response matching the CLARIFY_PROMPT_TEMPLATE schema, with contextually relevant mock questions).

4. Add a test in `tests/test_workflow.py` that verifies the ensemble produces deduplicated questions.

**Constraints:**
- The LLM clarification must be opt-in via env var `LLM_CLARIFY_ENABLED` (default: "1" / enabled) so heuristic-only mode is always available.
- Must not break existing tests.

---

### Upgrade 4: Visual Evaluation Dashboard

**Goal:** Create a visual dashboard for the eval harness results.

**Requirements:**

1. Create `evals/dashboard.py` that:
   - Loads `last_results.json` (and optionally historical results if you store them).
   - Builds a Gradio Blocks app (separate from the main app) with these tabs:

   **Tab 1: Overview**
   - Overall pass rate (big number + progress bar).
   - Pass rate by category (bar chart).
   - Selector accuracy (% of times the scoring picked the best variant).

   **Tab 2: Scoring Analysis**
   - Distribution of faithfulness scores across all cases (histogram).
   - Distribution of hallucination scores (histogram).
   - Scatter plot: faithfulness vs. hallucination score, colored by pass/fail.
   - Current scoring weights displayed as a table.

   **Tab 3: Case Details**
   - Searchable/filterable table of all eval cases.
   - Click a row to expand: see the input notes, generated draft, rubric results, flagged sentences, scores.
   - Color-code: green for pass, red for fail, yellow for partial.

   **Tab 4: Comparison** (if historical data exists)
   - Line chart of pass rate over time / across eval runs.

2. Add a launch script or CLI entry point: `python -m evals.dashboard` opens the dashboard.

3. Use Gradio's native plotting (or `gr.Plot` with plotly/matplotlib) — do NOT add heavy BI dependencies.

4. Style the dashboard to match the main app's dark theme (reuse CSS variables or similar aesthetic).

**Constraints:**
- Must work with the current `last_results.json` structure. Extend the structure if needed but maintain backward compatibility.
- Add `plotly` or `matplotlib` to `requirements.txt` if used.
- The dashboard is a separate Gradio app, not merged into the main app.

---

### Upgrade 5: GitHub Actions CI + Docker

**Goal:** Containerize the app and add CI that runs on every push.

**Requirements:**

1. Create `Dockerfile`:
   - Base image: `python:3.11-slim`.
   - Copy requirements, install deps.
   - Copy source code.
   - Expose port 7860 (Gradio default).
   - CMD: `python app.py`.
   - Add a `.dockerignore` (exclude `.venv`, `__pycache__`, `.pytest_cache`, `.git`, `*.pyc`).
   - Multi-stage build is optional but appreciated.

2. Create `docker-compose.yml`:
   - Service `app` building from Dockerfile, mapping port 7860.
   - Optional `ollama` service pulling `ollama/ollama` image (so the full stack runs with `docker-compose up`).

3. Create `.github/workflows/ci.yml`:
   - Triggers: push to `main`, pull requests to `main`.
   - Jobs:
     - **lint**: Run `ruff check .` (add `ruff` to requirements.txt dev section or as a CI-only install).
     - **test**: Set up Python 3.11, install requirements, run `pytest tests/ -v`.
     - **eval**: Run `python -m evals.run_evals --mock` (ensure run_evals supports a `--mock` flag to use MockLLMClient, add it if missing).
     - **docker-build**: Build the Docker image (don't push, just verify it builds).
   - Use `actions/checkout@v4`, `actions/setup-python@v5`.
   - Cache pip dependencies for speed.

4. Update `evals/run_evals.py`:
   - Add `--mock` CLI flag that forces `MockLLMClient` (for CI where Ollama isn't available).
   - Add proper `argparse` CLI interface if not already present.

**Constraints:**
- CI must pass with MockLLMClient (no real LLM needed).
- Docker image should be < 2GB (use slim base, clean up pip cache).
- All existing tests must pass in CI.

---

## General Guidelines

- **Do not break existing functionality.** Every change should be backward-compatible. If sentence-transformers or cross-encoder aren't installed, the app should still work with fallbacks.
- **Maintain the existing code style.** The project uses: snake_case, type hints, Pydantic models, module-level singletons, env vars for feature flags.
- **Update `requirements.txt`** with all new dependencies (pinned or unpinned, matching the existing style which is unpinned).
- **Update `README.md`** with: new features, how to run the dashboard, Docker instructions, CI badge placeholder.
- **Keep modules focused.** Each new file should be 50-100 lines, not monolithic.
- **Add tests** for new scoring modules (at minimum: test that FaithfulnessScorer returns a float 0-1, test that HallucinationDetector returns the expected dict structure, test the ensemble deduplication).
