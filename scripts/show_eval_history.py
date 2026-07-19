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

    print(f"{'Timestamp':<20} {'Judge':<10} {'Generator':<28} {'Faith.':>7} {'Rel.':>7} {'Recall':>7}  Gate")
    for e in entries:
        scores = e["scores"]
        gate = "PASS" if e["gate_passed"] else "FAIL"
        print(f"{e['timestamp']:<20} {e['judge']:<10} {e['generator']:<28} "
              f"{scores['faithfulness']:>7.3f} {scores['answer_relevancy']:>7.3f} "
              f"{scores['context_recall']:>7.3f}  {gate}")
