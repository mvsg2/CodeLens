"""RAGAS-based answer-quality eval — the second half of Aspect 4.

Unlike scripts/run_eval.py (free, retrieval-only), this makes real LLM
calls: one to generate each answer, plus several more internally for
RAGAS's own judge model. Real cost, real latency — run this before a
release or deploy, not on every commit. Use --limit while iterating to
bound the cost.

The judge model (the one grading answer quality, separate from gpt-4o
which always generates the answers) is selectable -- see app/eval.py's
JUDGE_MODELS and scripts/smoke_tests/judge_calibration.py for why this is
a deliberate choice, not a default to leave alone. qwen3-8b requires
OPENROUTER_API_KEY to be set; gpt-4o/gpt-5.2 use OPENAI_API_KEY.

Usage:
  python -m scripts.run_ragas_eval
  python -m scripts.run_ragas_eval --judge gpt-5.2
  python -m scripts.run_ragas_eval --judge qwen3-8b --limit 5
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from app.eval import evaluate_answers, check_answer_gate, ANSWER_QUALITY_THRESHOLDS, CONTEXT_RECALL_TARGET, JUDGE_MODELS, DEFAULT_JUDGE

REPO_ID = "fastapi__fastapi"
RESULTS_FILE = Path("data/ragas_eval_last_run.json")


def _json_safe(value):
    # df.to_dict(orient="records") doesn't fully convert every column to
    # plain Python types -- confirmed live: the "contexts" column (each
    # item's list of retrieved chunk texts) came through as a numpy
    # ndarray, which json.dump doesn't know how to serialize at all
    # (TypeError: Object of type ndarray is not JSON serializable) --
    # crashed the save step on a real CI run, after the eval itself had
    # already passed. Handles nested structures (a list containing
    # ndarrays or numpy scalars) recursively, not just the top level.
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):  # numpy scalar, e.g. np.float64
        return _json_safe(value.item())
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N golden-set items (cost control)")
    parser.add_argument("--judge", choices=list(JUDGE_MODELS), default=DEFAULT_JUDGE,
                        help=f"Which model grades answer quality (default: {DEFAULT_JUDGE})")
    args = parser.parse_args()

    print(f"Running RAGAS answer-quality eval for: {REPO_ID}")
    print(f"Judge model: {args.judge}")
    print("This calls the real LLM for every item — expect real cost and real latency.\n")

    result = evaluate_answers(REPO_ID, limit=args.limit, judge=args.judge)
    scores = result["scores"]

    print(f"Faithfulness:     {scores['faithfulness']:.3f}")
    print(f"Answer relevancy: {scores['answer_relevancy']:.3f}")
    print(f"Context recall:   {scores['context_recall']:.3f}")

    print("\nBy category:")
    for cat, cat_scores in result["by_category"].items():
        print(f"  {cat:<12} faithfulness={cat_scores['faithfulness']:.3f}  "
              f"answer_relevancy={cat_scores['answer_relevancy']:.3f}  "
              f"context_recall={cat_scores['context_recall']:.3f}")

    gate = check_answer_gate(scores)
    print(f"\nEVAL GATE: {'PASS' if gate else 'FAIL'}")
    for k, v in ANSWER_QUALITY_THRESHOLDS.items():
        status = "PASS" if scores.get(k, 0) >= v else "FAIL"
        print(f"  {k}: {scores.get(k, 0):.3f} (threshold {v}) [{status}]")

    # Reported, not gated -- see app/scoring.py's CONTEXT_RECALL_TARGET
    # comment for why (structurally unwinnable for the "negative" category,
    # noisy run-to-run on the rest).
    cr = scores.get("context_recall", 0)
    cr_status = "PASS" if cr >= CONTEXT_RECALL_TARGET else "FAIL"
    print(f"  context_recall: {cr:.3f} (target {CONTEXT_RECALL_TARGET}) [{cr_status}, report only]")

    # Every generated answer + its per-item scores, saved so a later "what
    # did it actually say" investigation (e.g. why answer_relevancy is
    # low) can read this instead of paying for a fresh set of LLM calls
    # just to look. Overwrites on each run -- last run's answers only, not
    # a history.
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    safe_results = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in result["results"]
    ]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"judge": args.judge, "scores": scores, "results": safe_results}, f, indent=2, ensure_ascii=False)
    print(f"\nPer-item answers + scores saved to {RESULTS_FILE}")

    raise SystemExit(0 if gate else 1)
