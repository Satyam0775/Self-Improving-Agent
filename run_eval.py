"""Run the full deterministic evaluation + self-improvement demonstration.

    python run_eval.py

No Groq API key or network access is needed. Exit codes:
    0  baseline failure found, fixed by an accepted improvement, no regressions
    2  the loop ran but ended in an unexpected outcome (rejection, regression, ...)
    1  unexpected error in the evaluation itself
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from app.config import IMPROVEMENTS_DIR, REPORTS_DIR, SCENARIOS_FILE, load_settings
from evals.improvement_loop import GroqProposer, run_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic evaluation and improvement loop.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_FILE)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--improvements-dir", type=Path, default=IMPROVEMENTS_DIR)
    parser.add_argument(
        "--proposer",
        choices=["template", "groq"],
        default="template",
        help="who drafts the improvement (default: deterministic template; 'groq' is optional and needs a key)",
    )
    args = parser.parse_args(argv)

    try:
        groq = None
        if args.proposer == "groq":
            settings = load_settings()
            if not settings.has_api_key:
                print("--proposer groq needs GROQ_API_KEY in .env; falling back to the deterministic template.")
            else:
                from app.groq_backend import make_client

                groq = GroqProposer(make_client(settings.groq_api_key), settings.groq_model)
        result = run_demo(args.scenarios, args.reports_dir, args.improvements_dir, args.proposer, groq)
        return int(result["exit_code"])
    except Exception:
        print("Evaluation failed with an unexpected error:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
