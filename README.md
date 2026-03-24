# Draft-to-Ready Writing Agent

Turn messy notes into a polished draft (and a self-check) using a local LLM via Ollama.

## Features
- Web UI where you paste notes and choose purpose/tone/length
- Agent workflow:
  - Clarification gate (heuristic + LLM ensemble)
  - Multi-variant draft generation with best-of-N selection
  - BERTScore-based faithfulness scoring
  - NLI-based hallucination detection
  - Lightweight rubric self-check
  - Finalize/edit pass (when Ollama is available)
- Supports multiple channels: `Email`, `WhatsApp`, `Microsoft Teams`
- Optional size control: `Small/Medium/Large` approximate length
- Advanced generation controls (temperature, top_p, top_k, repetition/frequency/presence penalties)
- Visual evaluation dashboard

## Folder structure
- `app.py` - Gradio web UI entrypoint
- `src/agent/workflow.py` - agent workflow orchestration
- `src/scoring/faithfulness.py` - BERTScore-based faithfulness scorer
- `src/scoring/hallucination.py` - NLI-based hallucination detector
- `src/llm/ollama_client.py` - calls local Ollama HTTP API
- `src/llm/mock_client.py` - fallback so UI works even without Ollama
- `src/prompts/templates.py` - prompt templates
- `src/schemas/models.py` - request/response models
- `evals/dashboard.py` - visual evaluation dashboard
- `evals/run_evals.py` - evaluation harness

## Setup
1. Install Python 3.10+ (Windows).
2. Create a virtual environment:
   - `python -m venv .venv`
3. Activate it:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
4. Install dependencies:
   - `pip install -r requirements.txt`

### Run (mock mode)
Even without Ollama, the app will start using a mock client.
Run:
- `python app.py`
Then open the local URL shown in the terminal.

### Enable real Ollama
1. Install Ollama.
2. Pull a model, for example:
   - `ollama pull mistral:7b-instruct-q4_0`
3. Start the web UI and set:
   - `setx OLLAMA_MODEL "mistral:7b-instruct-q4_0"`
4. Re-open the terminal and run:
   - `python app.py`

## Docker
Run the full stack with Docker Compose:
```bash
docker-compose up
```
This starts both the app (port 7860) and Ollama (port 11434).

To build just the app image:
```bash
docker build -t draft-to-ready .
```

## Evaluation harness
Run deterministic checks (default uses mock client):
- `python evals/run_evals.py`
- `python evals/run_evals.py --mock`  (explicit mock, useful for CI)
- `python evals/run_evals.py --cases evals/cases_hard.json`

Run with local Ollama model:
- `python evals/run_evals.py --use-ollama --model mistral:7b-instruct-q4_0`

Outputs:
- Pass/fail summary in terminal
- Detailed JSON results at `evals/last_results.json`

## Visual evaluation dashboard
Launch the dashboard to explore eval results:
```bash
python -m evals.dashboard
```
Opens on port 7861. Includes:
- Overview tab with pass rates and category breakdown
- Scoring analysis with current weights
- Case details viewer
- Historical comparison (placeholder for future runs)

## Calibrate heuristic selector weights
- `python evals/calibrate_scoring_weights.py`

## Environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral:7b-instruct-q4_0` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENT_JUDGE_ENABLED` | `0` | Enable LLM-based draft judge |
| `LLM_CLARIFY_ENABLED` | `1` | Enable LLM-based clarification ensemble |
| `HALLUCINATION_THRESHOLD` | `0.3` | Hallucination score threshold for warning |

## CI
<!-- CI badge placeholder: ![CI](https://github.com/YOUR_USER/YOUR_REPO/actions/workflows/ci.yml/badge.svg) -->
GitHub Actions runs on every push/PR to main:
- **lint**: ruff check
- **test**: pytest
- **eval**: mock eval harness
- **docker-build**: verify Docker image builds
