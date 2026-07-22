"""Post-deploy smoke test: shallow, fast checks that the running API is alive
and its critical paths respond correctly. Not a substitute for the eval gate
(scripts/run_eval.py) or integration tests — just "did the deploy work at all."

Assumes a repo has already been indexed (data/chroma/<repo_id> exists) and
the API container is up, e.g. via:
  docker compose up --build -d

Usage:
  python -m scripts.smoke_tests.deployment_smoke_test
  python -m scripts.smoke_tests.deployment_smoke_test --base-url http://localhost:8000
"""
import argparse
import sys
import time

import requests

REPO_URL = "https://github.com/fastapi/fastapi"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def wait_for_health(session: requests.Session, base_url: str, timeout: int = 150) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = session.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    # One shared session for every request in this run, instead of each call
    # implicitly opening/closing its own — works around an intermittent
    # Windows-specific requests/urllib3 hang seen with the default
    # one-session-per-call behavior (isolated via curl, which never hit it).
    session = requests.Session()

    results = []

    print(f"Waiting for {base}/health to come up...")
    up = wait_for_health(session, base)
    results.append(check("Server becomes healthy within 150s", up))
    if not up:
        print("\nServer never came up — aborting remaining checks.")
        sys.exit(1)

    r = session.get(f"{base}/health", timeout=5)
    results.append(check("GET /health returns 200", r.status_code == 200, f"got {r.status_code}"))
    results.append(check("GET /health body is {'status': 'ok'}", r.json() == {"status": "ok"}, str(r.json())))

    # Code query, no LLM call — proves retrieval + Chroma + rerank work end to end
    # inside the container without spending any tokens.
    r = session.post(f"{base}/query", json={
        "repo_url": REPO_URL,
        "question": "How does dependency injection work?",
        "include_answer": False,
    }, timeout=30)
    results.append(check("POST /query (code, no-llm) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"))
    if r.status_code == 200:
        body = r.json()
        sources = body.get("sources", [])
        results.append(check("  sources is non-empty", len(sources) > 0))
        results.append(check("  answer is null (include_answer=False)", body.get("answer") is None))
        results.append(check("  all sources are .py files", all(s["file"].endswith(".py") for s in sources),
                              str([s["file"] for s in sources])))

    # Doc query, no LLM call — proves the source_type filter works inside the container.
    r = session.post(f"{base}/query", json={
        "repo_url": REPO_URL,
        "question": "How do I install FastAPI?",
        "source_type": "doc",
        "include_answer": False,
    }, timeout=30)
    results.append(check("POST /query (doc, no-llm) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"))
    if r.status_code == 200:
        sources = r.json().get("sources", [])
        results.append(check("  sources is non-empty", len(sources) > 0))
        results.append(check("  all sources are .md files", all(s["file"].endswith(".md") for s in sources),
                              str([s["file"] for s in sources])))

    session.close()

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
