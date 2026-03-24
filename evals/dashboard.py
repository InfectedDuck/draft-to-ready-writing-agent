"""
Visual evaluation dashboard for the Draft-to-Ready Writing Agent.

Launch: python -m evals.dashboard
Opens on port 7861 to avoid conflicting with the main app (7860).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "last_results.json"
WEIGHTS_PATH = ROOT / "scoring_weights.json"

DARK_CSS = """
body { background: #09090b; color: #fafafa; font-family: 'Inter', system-ui, sans-serif; }
.gradio-container { background: transparent !important; }
h1, h2, h3 { color: #fafafa !important; }
"""


def load_results() -> Dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {"summary": {}, "results": []}
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_weights() -> Dict[str, float]:
    if not WEIGHTS_PATH.exists():
        return {}
    with WEIGHTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_overview_plot(data: Dict[str, Any]):
    """Bar chart of category pass rates."""
    if go is None:
        return None
    summary = data.get("summary", {})
    breakdown = summary.get("category_breakdown", {})
    if not breakdown:
        return None

    categories = list(breakdown.keys())
    rates = []
    for cat in categories:
        vals = breakdown[cat]
        total = vals.get("total", 0)
        passed = vals.get("passed", 0)
        rates.append(100.0 * passed / max(total, 1))

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=rates,
            marker_color=["#22c55e" if r >= 80 else "#eab308" if r >= 50 else "#ef4444" for r in rates],
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Pass Rate by Category",
        yaxis_title="Pass Rate (%)",
        yaxis_range=[0, 110],
        template="plotly_dark",
        paper_bgcolor="#09090b",
        plot_bgcolor="#18181b",
        font=dict(family="Inter, system-ui, sans-serif", color="#fafafa"),
        height=380,
    )
    return fig


def _make_case_table(data: Dict[str, Any]) -> List[List[str]]:
    """Build rows for the case details table."""
    results = data.get("results", [])
    rows = []
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        issues = "; ".join(r.get("issues", [])) or "-"
        rows.append([
            r.get("id", "?"),
            status,
            str(r.get("questions_count", 0)),
            str(r.get("has_draft", False)),
            str(r.get("has_final", False)),
            r.get("rubric_status", "?"),
            issues,
        ])
    return rows


def _make_selector_stats(data: Dict[str, Any]) -> str:
    """Compute selector accuracy summary."""
    results = data.get("results", [])
    best_passing = [
        r["selector_meta"]["selection_is_best_passing"]
        for r in results
        if r.get("selector_meta", {}).get("selection_is_best_passing") is not None
    ]
    if not best_passing:
        return "No selector data available."
    acc = 100.0 * sum(1 for x in best_passing if x) / len(best_passing)
    return f"Selector picked the best-passing variant **{acc:.1f}%** of the time ({len(best_passing)} cases evaluated)."


def _make_weights_table(weights: Dict[str, float]) -> List[List[str]]:
    return [[k, f"{v:.2f}"] for k, v in sorted(weights.items())]


def _case_detail_text(data: Dict[str, Any], case_id: str) -> str:
    """Return formatted detail for a specific case."""
    for r in data.get("results", []):
        if r.get("id") == case_id:
            return json.dumps(r, indent=2, default=str)
    return "Case not found."


def main() -> None:
    data = load_results()
    weights = load_weights()
    summary = data.get("summary", {})

    pass_rate = summary.get("pass_rate", 0)
    passed = summary.get("passed_cases", 0)
    total = summary.get("total_cases", 0)
    client = summary.get("client", "unknown")

    with gr.Blocks(title="Eval Dashboard", css=DARK_CSS) as dashboard:
        gr.Markdown(f"# Evaluation Dashboard\n**Client:** {client} | **Last run:** {passed}/{total} passed")

        with gr.Tabs():
            # ---- Tab 1: Overview ----
            with gr.Tab("Overview"):
                gr.Markdown(f"## Overall Pass Rate: **{pass_rate:.1f}%**")
                overview_plot = _make_overview_plot(data)
                if overview_plot:
                    gr.Plot(value=overview_plot)
                else:
                    gr.Markdown("*Install plotly for charts: `pip install plotly`*")
                gr.Markdown(_make_selector_stats(data))

            # ---- Tab 2: Scoring Analysis ----
            with gr.Tab("Scoring Analysis"):
                gr.Markdown("## Current Scoring Weights")
                if weights:
                    gr.Dataframe(
                        value=_make_weights_table(weights),
                        headers=["Weight", "Value"],
                        interactive=False,
                    )
                else:
                    gr.Markdown("*No scoring_weights.json found.*")

                gr.Markdown("## Scoring Distribution")
                gr.Markdown(
                    "*Faithfulness and hallucination score histograms will appear here "
                    "once eval results include those metrics (requires sentence-transformers).*"
                )

            # ---- Tab 3: Case Details ----
            with gr.Tab("Case Details"):
                gr.Markdown("## All Eval Cases")
                case_rows = _make_case_table(data)
                gr.Dataframe(
                    value=case_rows,
                    headers=["ID", "Status", "Questions", "Has Draft", "Has Final", "Rubric", "Issues"],
                    interactive=False,
                )

                gr.Markdown("## Case Detail Viewer")
                case_ids = [r.get("id", "?") for r in data.get("results", [])]
                case_dropdown = gr.Dropdown(choices=case_ids, label="Select case")
                detail_box = gr.Textbox(label="Case details (JSON)", lines=20, interactive=False)

                def show_detail(case_id):
                    return _case_detail_text(data, case_id)

                case_dropdown.change(fn=show_detail, inputs=[case_dropdown], outputs=[detail_box])

            # ---- Tab 4: Comparison ----
            with gr.Tab("Comparison"):
                gr.Markdown(
                    "## Historical Comparison\n\n"
                    "*To enable comparison over time, save each eval run's results "
                    "to a timestamped file (e.g., `results_2026-03-24.json`) in the `evals/` directory. "
                    "A future update will auto-detect and chart these.*"
                )

    dashboard.launch(server_port=7861)


if __name__ == "__main__":
    main()
