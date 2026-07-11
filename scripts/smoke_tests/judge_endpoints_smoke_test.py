"""Connectivity smoke test for every registered judge endpoint and the
answer_relevancy embedding model -- "can we reach it and get a real
response back at all," not "is its output good" (that's
judge_calibration.py's job).

Cheap and fast on purpose: one "reply with exactly: OK" chat call per
judge, one embed_query call for the embeddings model. Meant to be run
before anything that spends real money on a full golden-set run, so a
misconfigured API key / wrong base_url / wrong model-name string fails in
a couple seconds instead of partway through a 25-item run.

Covers exactly the ad hoc checks used while wiring up each new judge in
this project (OpenAI, OpenRouter, ASU Research Computing) -- written up
here as a permanent script instead of one-off inline commands, so the same
check can be rerun any time an API key rotates, a model gets deprecated,
or a new endpoint is added to app/eval.py's JUDGE_MODELS.

Usage:
  python -m scripts.smoke_tests.judge_endpoints_smoke_test
  python -m scripts.smoke_tests.judge_endpoints_smoke_test --judges gemma-4-31b,qwen3-8b
"""
import argparse
import sys

from app.eval import JUDGE_MODELS, build_judge, build_answer_relevancy_embeddings

PROBE_PROMPT = "Reply with exactly: OK"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def check_judge(judge_name: str) -> bool:
    cfg = JUDGE_MODELS[judge_name]
    label = f"judge '{judge_name}' ({cfg['model']} via {cfg['base_url'] or 'api.openai.com'})"
    try:
        judge = build_judge(judge_name)
        resp = judge.langchain_llm.invoke(PROBE_PROMPT)
        return check(label, bool(resp.content), f"response: {resp.content!r}")
    except Exception as e:
        return check(label, False, f"{type(e).__name__}: {str(e)[:200]}")


def check_embeddings() -> bool:
    from app.eval import EMBEDDING_MODEL, EMBEDDING_BASE_URL
    label = f"answer_relevancy embeddings ({EMBEDDING_MODEL} via {EMBEDDING_BASE_URL})"
    try:
        emb = build_answer_relevancy_embeddings()
        vec = emb.embeddings.embed_query("test query")
        return check(label, len(vec) > 0, f"vector length: {len(vec)}")
    except Exception as e:
        return check(label, False, f"{type(e).__name__}: {str(e)[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default=",".join(JUDGE_MODELS),
                        help="Comma-separated judge names to check (default: all registered)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip the answer_relevancy embeddings check")
    args = parser.parse_args()
    judge_names = [j.strip() for j in args.judges.split(",")]

    print("Checking every judge endpoint responds -- not testing output quality,")
    print("just reachability, auth, and correct model-name/base_url wiring.\n")

    results = [check_judge(name) for name in judge_names]
    if not args.skip_embeddings:
        results.append(check_embeddings())

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} endpoints reachable")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
