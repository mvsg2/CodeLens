"""RAGAS-based answer-quality eval — the second half of Aspect 4.

Unlike scripts/run_eval.py (free, retrieval-only), this makes real LLM
calls: one to generate each answer, plus several more internally for
RAGAS's own judge model. Real cost, real latency — run this before a
release or deploy, not on every commit. Use --limit while iterating to
bound the cost.

Usage:
  python -m scripts.run_ragas_eval
  python -m scripts.run_ragas_eval --limit 5
"""
import argparse

from app.eval import evaluate_answers, check_answer_gate, ANSWER_QUALITY_THRESHOLDS

REPO_ID = "fastapi__fastapi"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N golden-set items (cost control)")
    args = parser.parse_args()

    print(f"Running RAGAS answer-quality eval for: {REPO_ID}")
    print("This calls the real LLM for every item — expect real cost and real latency.\n")

    result = evaluate_answers(REPO_ID, limit=args.limit)
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
