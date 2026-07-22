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
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.eval import evaluate_answers, check_answer_gate, ANSWER_QUALITY_THRESHOLDS, JUDGE_MODELS, DEFAULT_JUDGE
from app.retrieval import GENERATOR_MODEL_NAME
from app.sourcing import upload_to_s3

REPO_ID = "fastapi__fastapi"
RESULTS_FILE = Path("data/ragas_eval_last_run.json")


def _upload_history_entry(entry: dict, timestamp: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(entry, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    try:
        upload_to_s3(tmp_path, f"eval-history/{REPO_ID}/{timestamp}.json")
    finally:
        tmp_path.unlink(missing_ok=True)


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

    # Captured once, at the start of the attempt -- reused for both the
    # success and crash paths below, so a history entry's timestamp means
    # "when this run happened," not "when it happened to finish" (some
    # runs take 8+ minutes; start time is the more meaningful anchor).
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    try:
        result = evaluate_answers(REPO_ID, limit=args.limit, judge=args.judge)
    except Exception as e:
        # Previously, a mid-run crash (a real, recurring failure mode this
        # project has hit repeatedly -- 402s, 429s, None-choices) meant
        # NO history entry ever got written at all: everything below this
        # point, including the S3 upload, only ran after evaluate_answers()
        # returned successfully. "The run crashed" and "the run never ran"
        # were indistinguishable in the history -- there was no way to see
        # an error rate over time, only silence. Recording a minimal entry
        # here closes that gap, then re-raises so CI still fails loudly,
        # exactly as before.
        print(f"\nCRASHED: {type(e).__name__}: {e}")
        crash_entry = {
            "timestamp": timestamp,
            "commit": os.environ.get("CODELENS_PIN_COMMIT"),
            "judge": args.judge,
            "generator": GENERATOR_MODEL_NAME,
            "crashed": True,
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],
        }
        try:
            _upload_history_entry(crash_entry, timestamp)
            print(f"Crash recorded to eval-history/{REPO_ID}/{timestamp}.json")
        except Exception as upload_err:
            print(f"(also failed to record the crash to S3: {upload_err})")
        traceback.print_exc()
        raise

    scores = result["scores"]

    print(f"Faithfulness:     {scores['faithfulness']:.3f}")
    print(f"Answer relevancy: {scores['answer_relevancy']:.3f}")

    gen_usage = result["generator_usage"]
    judge_usage = result["judge_usage"]
    timing = result["timing"]

    print(f"\nTiming: generation={timing['generation_seconds']:.1f}s  "
          f"judging={timing['judging_seconds']:.1f}s  "
          f"total={timing['total_seconds']:.1f}s")

    def _fmt_cost(usage):
        return f"${usage['estimated_cost_usd']:.4f}" if usage["estimated_cost_usd"] is not None else "unknown (no verified pricing for this model)"

    print(f"\nGenerator tokens: {gen_usage['total_tokens']:,} "
          f"({gen_usage['prompt_tokens']:,} prompt + {gen_usage['completion_tokens']:,} completion) "
          f"-- {_fmt_cost(gen_usage)}")
    print(f"Judge tokens:     {judge_usage['total_tokens']:,} "
          f"({judge_usage['prompt_tokens']:,} prompt + {judge_usage['completion_tokens']:,} completion) "
          f"-- {_fmt_cost(judge_usage)}")

    print("\nBy category:")
    for cat, cat_scores in result["by_category"].items():
        print(f"  {cat:<12} faithfulness={cat_scores['faithfulness']:.3f}  "
              f"answer_relevancy={cat_scores['answer_relevancy']:.3f}")

    gate = check_answer_gate(scores)
    print(f"\nEVAL GATE: {'PASS' if gate else 'FAIL'}")
    for k, v in ANSWER_QUALITY_THRESHOLDS.items():
        status = "PASS" if scores.get(k, 0) >= v else "FAIL"
        print(f"  {k}: {scores.get(k, 0):.3f} (threshold {v}) [{status}]")

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
        json.dump({
            "judge": args.judge,
            "scores": scores,
            "generator_usage": gen_usage,
            "judge_usage": judge_usage,
            "timing": timing,
            "results": safe_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nPer-item answers + scores saved to {RESULTS_FILE}")

    # Score history -- data/ragas_eval_last_run.json above only ever holds
    # the most recent run, and doesn't survive a GitHub-hosted runner being
    # destroyed after the job ends anyway. Uploading one small object per
    # run to S3 (real durable storage, not the ephemeral runner disk --
    # reuses app.sourcing's existing upload_to_s3 helper, same bucket
    # manifests already go to) is what actually makes "did this get
    # better or worse over time" answerable later, instead of only ever
    # having whatever happens to still be visible in a CI log.
    history_entry = {
        "timestamp": timestamp,
        "commit": os.environ.get("CODELENS_PIN_COMMIT"),
        "judge": args.judge,
        "generator": GENERATOR_MODEL_NAME,
        "golden_set_size": len(safe_results),
        "scores": scores,
        "gate_passed": gate,
        "crashed": False,
        "generator_usage": gen_usage,
        "judge_usage": judge_usage,
        "timing": timing,
    }
    _upload_history_entry(history_entry, timestamp)

    raise SystemExit(0 if gate else 1)
