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

from app.eval import evaluate_answers, check_answer_gate, ANSWER_QUALITY_THRESHOLDS, CONTEXT_RECALL_TARGET, JUDGE_MODELS, DEFAULT_JUDGE

REPO_ID = "fastapi__fastapi"

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

    raise SystemExit(0 if gate else 1)
