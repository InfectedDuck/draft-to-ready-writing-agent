import json
import itertools
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals" / "cases_hard.json"
WEIGHTS_PATH = ROOT / "evals" / "scoring_weights.json"


def run_eval_capture() -> tuple[int, int, float]:
    """
    Runs evals in mock mode and extracts:
    - passed_cases
    - total_cases
    - selector_best_passing_accuracy
    """
    cmd = [sys.executable, str(ROOT / "evals" / "run_evals.py"), "--cases", str(CASES)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")

    passed = None
    total = None
    sel_acc = None
    for line in out.splitlines():
        if line.startswith("Passed:"):
            # Passed: 6/6 (100.0%)
            parts = line.replace("Passed:", "").strip().split()
            first = parts[0]
            if "/" in first:
                a, b = first.split("/", 1)
                try:
                    passed = int(a)
                    total = int(b)
                except Exception:
                    pass
        if line.startswith("Selector best-passing accuracy:"):
            # Selector best-passing accuracy: 100.0% (6 cases evaluated)
            try:
                sel_acc_str = line.split(":")[1].strip().split("%")[0].strip()
                sel_acc = float(sel_acc_str)
            except Exception:
                pass

    if passed is None or total is None or sel_acc is None:
        raise RuntimeError("Failed to parse eval output. Run evals manually to inspect.")

    return passed, total, sel_acc


def main() -> None:
    # Small grid search over two key weights.
    base = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))

    w_next_options = [0.6, 1.0, 1.4, 2.0]
    w_intent_options = [0.6, 1.0, 1.4, 2.0]

    best = None
    best_weights = None

    for w_next, w_intent in itertools.product(w_next_options, w_intent_options):
        weights = dict(base)
        weights["w_next_step"] = float(w_next)
        weights["w_intent"] = float(w_intent)
        WEIGHTS_PATH.write_text(json.dumps(weights, indent=2), encoding="utf-8")

        passed, total, sel_acc = run_eval_capture()
        # Primary objective: selector accuracy; tie-breaker: pass rate.
        metric = (sel_acc, passed / max(total, 1))
        if best is None or metric > best:
            best = metric
            best_weights = weights

        print(f"w_next={w_next}, w_intent={w_intent} => selector_acc={sel_acc}%, pass={passed}/{total}")

    if best_weights is not None:
        WEIGHTS_PATH.write_text(json.dumps(best_weights, indent=2), encoding="utf-8")
    print("\nBest weights saved to scoring_weights.json")
    print(best_weights)


if __name__ == "__main__":
    main()

