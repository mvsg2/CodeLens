import argparse
from app.retrieval import answer_query

REPO_ID = "fastapi__fastapi"

QUERIES = [
    "Where is request routing implemented?",
    "How does dependency injection work?",
    "Where is authentication handled?",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM call, print retrieved sources only")
    args = parser.parse_args()

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        result = answer_query(query, REPO_ID, include_answer=not args.no_llm)

        if result["answer"] is not None:
            print(f"\nANSWER:\n{result['answer']}")

        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  {s['file']}:{s['line']} ({s['function']})")
