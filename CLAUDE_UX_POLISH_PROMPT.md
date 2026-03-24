# UX Polish & Code Cleanup Prompt (Plan Mode → Agent Mode)

## Instructions for Claude

**Start in Plan Mode.** Before writing any code, create a detailed implementation plan covering all 10 tasks below. The plan should:
- List every file that needs to be read, modified, or created for each task
- Identify dependencies between tasks (e.g., removing the Language field in Task 8 affects schemas, workflow, templates, mock_client, tests, evals, and app.py — plan the order)
- Flag any potential conflicts between tasks (e.g., Task 2 streaming changes the handler functions that Task 3 and Task 4 also modify)
- Propose the safest order of execution to avoid breaking things
- For Task 6 (code cleanup), list what you find before deleting anything

Once I approve the plan, switch to Agent Mode and execute it.

---

## Project Context

I have a "Draft-to-Ready Writing Agent" project — a Gradio 6 app that turns messy notes into polished messages using LLMs. The codebase structure:

```
app.py                              # Gradio 6 UI (~1837 lines) with landing page + app page
src/
  schemas/models.py                 # DraftRequest, DraftResponse (Pydantic)
  agent/workflow.py                 # Core agent: clarification → draft → scoring → rubric → finalize
  prompts/templates.py              # DRAFT_PROMPT_TEMPLATE, FINALIZE_PROMPT_TEMPLATE, CLARIFY_PROMPT_TEMPLATE, JUDGE_DRAFT_PROMPT_TEMPLATE
  scoring/faithfulness.py           # BERTScore-based faithfulness scorer (sentence-transformers)
  scoring/hallucination.py          # NLI-based hallucination detector (cross-encoder)
  scoring/__init__.py               # Package marker
  llm/ollama_client.py              # OllamaClient (HTTP to local Ollama)
  llm/openrouter_client.py          # OpenRouterClient (cloud API, OpenAI-compatible)
  llm/mock_client.py                # MockLLMClient (deterministic stub for tests/fallback)
evals/
  run_evals.py                      # Batch eval harness with --mock flag
  dashboard.py                      # Visual eval dashboard (separate Gradio app, port 7861)
  calibrate_scoring_weights.py      # Grid search weight tuning
  cases.json, cases_hard.json       # Eval cases
  scoring_weights.json              # Calibrated weights
tests/
  test_workflow.py                  # Pytest tests
.env                                # OPENROUTER_API_KEY (gitignored)
.gitignore, .dockerignore
Dockerfile, docker-compose.yml
.github/workflows/ci.yml
requirements.txt                    # gradio, pydantic, requests, python-dotenv, pytest, sentence-transformers, plotly
README.md
```

Key details:
- LLM provider is switchable via a UI dropdown: OpenRouter (cloud), Ollama (local), Mock (demo)
- The app uses python-dotenv to load .env
- The app has a landing page (hero section + features + how-it-works) and an app page (30/70 layout)
- CUSTOM_CSS is passed to demo.launch(css=CUSTOM_CSS, server_name="0.0.0.0")

---

## Tasks (ALL 10 must be completed)

### Task 1: Rewrite README.md as a portfolio-grade document
This project is being submitted to CUHK (Chinese University of Hong Kong) for a graduate application. The README must tell a compelling story.

- Start with a one-line hook, then a 2-3 sentence summary of what the project does and why it matters
- Add an architecture diagram using a Mermaid code block showing the full pipeline: User Input → Clarification Gate (Heuristic + LLM Ensemble) → Multi-Variant Draft Generation → BERTScore Faithfulness + NLI Hallucination Detection → Rubric Self-Check → Finalize
- Add a "Research References" section citing: Zhang et al. 2020 (BERTScore: Evaluating Text Generation with BERT), Honovich et al. 2022 (TRUE: Re-evaluating Factual Consistency Evaluation), and the concept of ensemble clarification
- Add a "Key Design Decisions" section explaining: why ensemble (heuristic + LLM) for clarification, why BERTScore over keyword matching, why NLI for hallucination detection, why best-of-N selection with heuristic+judge
- Keep the existing Setup, Docker, Eval, CI sections but clean them up and make them scannable
- Add a "Live Demo" section with placeholder text: "Hugging Face Spaces deployment coming soon"
- Add a "Supported LLM Providers" section listing OpenRouter, Ollama, Mock with one-line descriptions
- Add badges at the top (Python 3.11, Gradio 6, Docker, License MIT, etc.) using shield.io markdown badges

### Task 2: Add streaming output for draft generation
- When the user clicks "Generate Draft", the draft text should stream in word-by-word instead of appearing all at once after a long wait
- Use Gradio's built-in streaming (yield from the handler function)
- This only needs to work for the draft output textbox, not all outputs
- The streaming should work for OpenRouter (which supports streaming via SSE) and gracefully fall back to non-streaming for Ollama and Mock
- Update src/llm/openrouter_client.py to add a generate_stream() method that yields text chunks as they arrive from the API
- In app.py, make the generate draft button handler yield intermediate updates so the user sees text appearing progressively

### Task 3: Add proper error handling with user-visible feedback
- If OpenRouter returns a 401 (bad API key), show a clear Gradio warning: "Invalid OpenRouter API key. Check your .env file or switch to Mock mode."
- If OpenRouter returns a 429 (rate limit), show: "Rate limit reached. Wait a moment and try again."
- If Ollama is not running and user selects Ollama, show: "Ollama is not running. Start it with 'ollama serve' or switch to OpenRouter/Mock."
- Do NOT silently fall back to Mock — the user should know what happened via gr.Warning() or gr.Info()
- Add a try/except around the LLM call in generate_draft that catches specific HTTP exceptions and shows the right message

### Task 4: Add input validation
- If raw_notes is empty when user clicks Generate, show a gr.Warning and return without calling the LLM
- If raw_notes has fewer than 10 characters, show a warning suggesting the user add more detail
- If OpenRouter is selected but OPENROUTER_API_KEY is not set in the environment, show a clear message telling the user to set it in .env or switch providers

### Task 5: Update requirements.txt
- Make sure all dependencies are listed (including python-dotenv which is actively used)
- Don't pin versions (match the existing unpinned style)

### Task 6: Code cleanup — find and fix dead code, unused imports, and errors
- Search the ENTIRE codebase for unused imports, unused functions, unreachable code, and dead code. Remove them.
- Check if any function or template is defined but never called. Flag and fix.
- Check that all Python packages under src/ have __init__.py files
- Make sure MockLLMClient properly handles all prompt types the workflow sends it
- Look for any subtle bugs: wrong variable names, off-by-one errors, unreachable branches, mismatched function signatures
- Report what you found and removed

### Task 7: Fix the light theme / force dark mode
- The app is designed for dark mode ONLY, but Gradio 6 can default to light mode based on the user's OS settings, which makes the UI completely unreadable (dark text on dark backgrounds, invisible elements, etc.)
- Force dark mode so the app always renders correctly regardless of system preference
- Use Gradio 6's theme parameter (e.g., gr.themes.Base() with dark mode) or inject CSS that overrides @media (prefers-color-scheme: light) to still use dark colors
- Verify that ALL text remains readable: labels, placeholders, dropdowns, buttons, accordions, rubric cards, step indicators

### Task 8: Remove non-functional features
- REMOVE the "Language" text input from the Context card on the app page. The app only generates in English and this field does nothing useful — it confuses users into thinking other languages work.
- Remove any "recording" or audio-related UI elements if they exist — they are not functional.
- IMPORTANT: After removing the Language field, you MUST update ALL references across the entire codebase:
  - src/schemas/models.py (DraftRequest model)
  - src/agent/workflow.py (any usage of req.language)
  - src/prompts/templates.py (remove {language} from prompt templates)
  - src/llm/mock_client.py (if it parses Language from prompts)
  - tests/test_workflow.py (remove language= from test DraftRequest calls)
  - evals/run_evals.py and evals/cases.json, evals/cases_hard.json (remove language from case inputs)
  - app.py (remove from build_request, generate_draft, all_inputs, handler functions)
- Make sure NOTHING breaks after removal. Every function signature, every input list, every handler must still match.

### Task 9: Fix landing page text alignment
- On the landing page, the hero heading (h1) and subtitle (.lp-hero-sub) appear LEFT-ALIGNED instead of centered
- This is likely because Gradio 6's default CSS overrides the inline text-align: center
- Fix it by adding !important to the text-align rules in LANDING_PAGE_HTML's inline styles, or add overrides in CUSTOM_CSS that target .lp-hero, .lp-hero h1, and .lp-hero-sub
- Also check that the features grid and how-it-works section are properly centered

### Task 10: Make the UI more user-friendly with better explanations and guidance
The app should feel intuitive to a first-time user (like a CUHK professor evaluating this project). Add concise helper text throughout:

- **"Your raw notes / bullets" textbox**: Use a realistic, detailed placeholder example like: "I need to email my professor Dr. Smith about extending the deadline for my CS101 assignment. Reason: I was sick last week. Original deadline was March 20."
- **"Your answers to questions" textbox**: Add a short description/subtitle: "After clicking Generate, the agent may ask clarifying questions. Type your answers here and click Generate again."
- **Purpose field**: Better placeholder with examples
- **Audience field**: Better placeholder with examples
- **Above or below the Generate Draft button**: Add a one-line workflow explanation: "The agent will ask clarifying questions → generate multiple draft variants → score and select the best one"
- **Draft Stage section**: Add a subtle one-liner: "The agent generates multiple variants and selects the highest-scoring draft based on faithfulness, tone, and format."
- **Finalization section**: Add: "Click Finalize Draft to run an editing pass that addresses issues found by the self-check rubric."
- Style all helper text in muted/subtle color (use var(--muted) or var(--subtle)) so it doesn't clutter the UI

---

## Critical Rules
- Read every file before editing it
- Don't break existing functionality — the app must work with OpenRouter, Ollama, and Mock after all changes
- After removing the Language field (Task 8), verify ALL files are updated. Missing one reference will crash the app.
- After changing handler function signatures (Tasks 2, 3, 4, 8), verify that all_inputs, all handler functions, and generate_draft all have matching parameter counts
- If tasks conflict with each other, resolve them in the plan phase before coding
