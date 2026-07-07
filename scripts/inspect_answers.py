"""Manual cross-check tool: generate real answers for golden-set queries and
print them next to the actual retrieved chunk text, so a human can eyeball
whether each claim in the answer is actually supported by what was
retrieved — no RAGAS judge involved, just raw generation + raw evidence.

This is what caught the RAGAS faithfulness score being unreliable for the
`identifier` category (see notes/eval-harness-and-ragas.md): the automated
score said 0.238, but every claim in every answer checked out against both
the retrieved chunks and independently-verified ground truth.

Costs one real LLM call per item (to generate the answer) — no RAGAS judge
calls, so much cheaper than scripts/run_ragas_eval.py.

Usage:
  python -m scripts.inspect_answers --category identifier
  python -m scripts.inspect_answers --category negative --full
  python -m scripts.inspect_answers --query "Where is authentication handled?"
  python -m scripts.inspect_answers --limit 3
"""
import argparse

from app.eval import GOLDEN_SET
from app.retrieval import answer_query

REPO_ID = "fastapi__fastapi"
PREVIEW_CHARS = 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO_ID, help="repo_id to query against")
    parser.add_argument("--category", default=None,
                        help="Only inspect items in this golden-set category "
                             "(localization, identifier, explanation, doc, negative, boundary)")
    parser.add_argument("--query", default=None,
                        help="Only inspect the item matching this exact query text")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of items inspected (cost control)")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text instead of a short preview")
    args = parser.parse_args()

    items = GOLDEN_SET
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.query:
        items = [i for i in items if i["query"] == args.query]
    if args.limit:
        items = items[:args.limit]

    if not items:
        print("No matching golden-set items — check --category/--query spelling.")
        raise SystemExit(1)

    print(f"Inspecting {len(items)} item(s) against repo: {args.repo}\n")

    for item in items:
        result = answer_query(
            item["query"], args.repo,
            source_type=item["source_type"],
            include_answer=True,
            include_context=True,
        )

        print("=" * 80)
        print(f"QUERY:    {item['query']}")
        print(f"CATEGORY: {item['category']}")
        print(f"EXPECTED: {item['expected_answer']}")
        print(f"\nGENERATED ANSWER:\n{result['answer']}")

        chunks = result["context_chunks"]
        print(f"\nRETRIEVED CONTEXT CHUNKS ({len(chunks)}):")
        for i, chunk in enumerate(chunks):
            print(f"--- chunk {i + 1} ({len(chunk)} chars) ---")
            print(chunk if args.full else chunk[:PREVIEW_CHARS])
        print()
