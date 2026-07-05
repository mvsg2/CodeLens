from app.eval import evaluate_retrieval

REPO_ID = "fastapi__fastapi"
HIT_RATE_THRESHOLD = 0.75  # mirrors the context-recall gate from Aspect 4

if __name__ == "__main__":
    print(f"Running retrieval eval for: {REPO_ID}\n")
    scores = evaluate_retrieval(REPO_ID, top_k=5)

    for r in scores["results"]:
        status = "HIT " if r["hit"] else ("LANG" if r["translation_miss"] else "MISS")
        print(f"[{status}] ({r['category']}/{r['source_type']}) {r['query']}")
        if not r["hit"]:
            print(f"       expected:  {r['expected']}")
            print(f"       retrieved: {r['retrieved']}")

    print(f"\nHit rate @5: {scores['hit_rate']:.3f}  (boundary excluded)")
    print(f"MRR @5:      {scores['mrr']:.3f}")
    print("By category:")
    for cat, rate in scores["by_category"].items():
        print(f"  {cat:<12} {rate:.3f}")

    gate = scores["hit_rate"] >= HIT_RATE_THRESHOLD
    print(f"\nEVAL GATE: {'PASS' if gate else 'FAIL'} (threshold {HIT_RATE_THRESHOLD})")
    raise SystemExit(0 if gate else 1)
