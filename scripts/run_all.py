"""End-to-end orchestrator: sourcing -> pipeline (encode) -> eval -> retrieval demo.

Re-encoding is the expensive step (GPU minutes over ~20k chunks), so it only
runs when something that invalidates the existing index has actually changed:
a new commit was pulled, PIPELINE_VERSION was bumped (chunking/metadata logic
changed), or the embedding model name changed. That state is tracked in
data/index_state/<repo_id>.json by app.pipeline.needs_reindex/mark_indexed.

Usage:
  python -m scripts.run_all
  python -m scripts.run_all --force-reencode
  python -m scripts.run_all --skip-eval --no-llm --query "How does dependency injection work?"
"""
import argparse
import sys

from app.sourcing import clone_repo, collect_files, build_manifest, save_manifest, upload_to_s3
from app.pipeline import run_pipeline, needs_reindex
from app.eval import evaluate_retrieval
from app.retrieval import answer_query

REPO_URL = "https://github.com/fastapi/fastapi"
HIT_RATE_THRESHOLD = 0.75  # matches scripts/run_eval.py's CI gate


def banner(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO_URL, help="GitHub repo URL")
    parser.add_argument("--force-reencode", action="store_true",
                        help="Re-run the pipeline even if nothing changed")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the retrieval-only eval gate")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM call in the retrieval demo (sources only)")
    parser.add_argument("--query", default="Where is request routing implemented?",
                        help="Question for the end-to-end retrieval demo")
    parser.add_argument("--require-gate", action="store_true",
                        help="Exit before the demo (and any LLM cost) if the eval gate fails")
    args = parser.parse_args()

    # ── 1. Sourcing ────────────────────────────────────
    banner("STEP 1/4 — SOURCING")
    repo_path = clone_repo(args.repo)
    files = collect_files(repo_path)
    manifest = build_manifest(args.repo, repo_path, files)
    repo_id = manifest["repo"].replace("/", "__")
    manifest_path = save_manifest(manifest, repo_id)
    upload_to_s3(manifest_path, f"repos/{repo_id}/manifest.json")
    print(f"Repo: {manifest['repo']} @ {manifest['commit_sha'][:7]} | {manifest['total_files']} files")

    # ── 2. Pipeline (encode) — only if stale ──────────
    banner("STEP 2/4 — PIPELINE (parse, chunk, embed, index)")
    if args.force_reencode or needs_reindex(repo_id, manifest):
        print("Index is stale or missing (new commit / pipeline version change) — re-encoding...")
        total = run_pipeline(repo_id)
        print(f"Indexed {total} chunks.")
    else:
        print("Index is up to date (same commit + pipeline version) — skipping re-encode.")

    # ── 3. Eval (retrieval-only, free) — gates the LLM-costing demo below ──
    if not args.skip_eval:
        banner("STEP 3/4 — EVAL (retrieval-only, no LLM cost)")
        scores = evaluate_retrieval(repo_id)
        print(f"Hit rate @5: {scores['hit_rate']:.3f}   MRR @5: {scores['mrr']:.3f}")
        print("By category:")
        for cat, rate in scores["by_category"].items():
            print(f"  {cat:<12} {rate:.3f}")

        gate_passed = scores["hit_rate"] >= HIT_RATE_THRESHOLD
        print(f"\nEVAL GATE: {'PASS' if gate_passed else 'FAIL'} (threshold {HIT_RATE_THRESHOLD})")
        if not gate_passed:
            print("WARNING: retrieval quality is below threshold — the demo answer below "
                  "may be built on bad sources.")
            if args.require_gate:
                print("--require-gate set: stopping before the retrieval demo (no LLM call made).")
                sys.exit(1)
    else:
        banner("STEP 3/4 — EVAL (skipped)")

    # ── 4. Retrieval demo (the actual user-facing path) ─
    banner("STEP 4/4 — RETRIEVAL DEMO")
    result = answer_query(args.query, repo_id, include_answer=not args.no_llm)
    print(f"QUERY: {args.query}")
    if result["answer"] is not None:
        print(f"\nANSWER:\n{result['answer']}")
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  {s['file']}:{s['line']} ({s['function']})")


if __name__ == "__main__":
    main()
