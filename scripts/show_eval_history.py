"""Print RAGAS eval score history over time -- the trend
scripts/run_ragas_eval.py writes one small entry for, per run, to S3
(eval-history/<repo_id>/<timestamp>.json), specifically so "did this get
better or worse" has real recorded data behind it instead of relying on
memory of past runs' printed logs.

Free to run -- lists/reads existing S3 objects, makes no LLM calls.

Usage:
  python -m scripts.show_eval_history
  python -m scripts.show_eval_history --repo fastapi__fastapi --limit 10
"""
import argparse
import json
import os

import boto3

from app.config import S3_BUCKET

REPO_ID = "fastapi__fastapi"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO_ID, help="repo_id to show history for")
    parser.add_argument("--limit", type=int, default=20, help="show at most the N most recent runs")
    args = parser.parse_args()

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", None),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    prefix = f"eval-history/{args.repo}/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    keys = sorted(obj["Key"] for obj in response.get("Contents", []))
    if not keys:
        print(f"No eval history found under s3://{S3_BUCKET}/{prefix}")
        raise SystemExit(0)

    keys = keys[-args.limit:]
    entries = []
    for key in keys:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        entries.append(json.loads(body))

    # .get() throughout -- older history entries predate context_recall
    # being dropped (app/scoring.py has the why), predate
    # generator_usage/judge_usage/timing being added, and predate crashed
    # runs getting recorded at all (previously a mid-run crash wrote no
    # entry whatsoever -- see run_ragas_eval.py's crash-handling comment).
    # Keeps this script working across all of those boundaries instead of
    # a KeyError on old entries.
    print(f"{'Timestamp':<20} {'Judge':<10} {'Generator':<28} {'Faith.':>7} {'Rel.':>7}  {'Cost':>10}  {'Time':>7}  Status")
    crashed_count = 0
    for e in entries:
        if e.get("crashed"):
            crashed_count += 1
            print(f"{e['timestamp']:<20} {e.get('judge', '?'):<10} {e.get('generator', '?'):<28} "
                  f"{'--':>7} {'--':>7}  {'--':>10}  {'--':>7}  "
                  f"CRASHED ({e.get('error_type', 'unknown error')})")
            continue

        scores = e["scores"]
        gate = "PASS" if e["gate_passed"] else "FAIL"
        gen_cost = (e.get("generator_usage") or {}).get("estimated_cost_usd")
        judge_cost = (e.get("judge_usage") or {}).get("estimated_cost_usd")
        if gen_cost is not None and judge_cost is not None:
            cost_str = f"${gen_cost + judge_cost:.4f}"
        else:
            cost_str = "n/a"
        total_seconds = (e.get("timing") or {}).get("total_seconds")
        time_str = f"{total_seconds:.0f}s" if total_seconds is not None else "n/a"
        print(f"{e['timestamp']:<20} {e['judge']:<10} {e['generator']:<28} "
              f"{scores['faithfulness']:>7.3f} {scores['answer_relevancy']:>7.3f}  "
              f"{cost_str:>10}  {time_str:>7}  {gate}")

    if crashed_count:
        print(f"\n{crashed_count}/{len(entries)} runs crashed outright ({crashed_count / len(entries) * 100:.0f}%)")
