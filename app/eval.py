"""Eval harness (Aspect 4) — golden test set + two scorers.

The golden set is ground truth: every expected_source was verified by hand
against the cloned repo at data/repos/fastapi__fastapi. Do NOT regenerate
entries from system output — that bakes retrieval bugs into the truth.

Two independent scorers sit on top of the same golden set:

- evaluate_retrieval(): retrieval only (hit rate / MRR on expected_sources).
  No LLM calls, free, fast — safe to run on every change.
- evaluate_answers(): RAGAS-based answer quality (faithfulness, answer
  relevancy, context recall). Calls the real LLM for every item, both to
  generate the answer and, internally, for RAGAS's own judge model — real
  cost, real latency. Run this far less often (e.g. before a release), not
  on every commit.
"""

from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

from app.retrieval import build_retriever, rerank, answer_query
from app.scoring import norm_path, is_translation_miss, ANSWER_QUALITY_THRESHOLDS, check_answer_gate  # noqa: F401

# ── Golden test set ───────────────────────────────────
# category: localization | identifier | explanation | doc | negative | boundary
# negative queries have no expected sources — correct behavior is "cannot answer";
# they are skipped by the retrieval scorer and reserved for the RAGAS faithfulness eval.

GOLDEN_SET = [
    # ── code: localization ────────────────────────────
    {
        "query": "Where is request routing implemented?",
        "expected_answer": "fastapi/routing.py — APIRouter.add_api_route creates an APIRoute and appends it to self.routes",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where is OAuth2 password bearer authentication implemented?",
        "expected_answer": "fastapi/security/oauth2.py — OAuth2PasswordBearer extracts the bearer token from the Authorization header",
        "expected_sources": ["fastapi/security/oauth2.py"],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where are WebSocket routes registered?",
        "expected_answer": "fastapi/routing.py — add_api_websocket_route creates an APIWebSocketRoute (also exposed on FastAPI in applications.py)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where are validation errors converted into HTTP responses?",
        "expected_answer": "fastapi/exception_handlers.py — request_validation_exception_handler returns a 422 JSONResponse",
        "expected_sources": ["fastapi/exception_handlers.py"],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where does FastAPI validate HTTP Basic auth credentials?",
        "expected_answer": "fastapi/security/http.py — HTTPBasic parses and decodes the Authorization header",
        "expected_sources": ["fastapi/security/http.py"],
        "source_type": "code",
        "category": "localization",
    },

    # ── code: exact identifiers (BM25 stress tests) ───
    {
        "query": "Where is `add_api_route` defined?",
        "expected_answer": "fastapi/routing.py (APIRouter.add_api_route) and fastapi/applications.py (FastAPI.add_api_route)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `solve_dependencies` implemented?",
        "expected_answer": "fastapi/dependencies/utils.py — async solve_dependencies resolves the dependency tree for a request",
        "expected_sources": ["fastapi/dependencies/utils.py"],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "What does `jsonable_encoder` do?",
        "expected_answer": "fastapi/encoders.py — recursively converts objects (models, dataclasses, datetimes) into JSON-compatible types",
        "expected_sources": ["fastapi/encoders.py"],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `serialize_response` implemented?",
        "expected_answer": "fastapi/routing.py — serialize_response validates and serializes the return value against the response model",
        "expected_sources": ["fastapi/routing.py"],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `include_router` defined?",
        "expected_answer": "fastapi/routing.py (APIRouter.include_router) and fastapi/applications.py (FastAPI.include_router)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "source_type": "code",
        "category": "identifier",
    },

    # ── code: explanation ─────────────────────────────
    {
        "query": "How does dependency injection resolve nested dependencies?",
        "expected_answer": "fastapi/dependencies/utils.py — solve_dependencies walks the dependant tree, awaiting sub-dependencies and caching results",
        "expected_sources": ["fastapi/dependencies/utils.py"],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does `Depends` declare a dependency for a path operation parameter?",
        "expected_answer": "fastapi/param_functions.py — Depends returns a params.Depends marker consumed during dependant analysis",
        "expected_sources": ["fastapi/param_functions.py"],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How are background tasks executed after a response is sent?",
        "expected_answer": "fastapi/background.py — BackgroundTasks collects tasks that Starlette runs after the response",
        "expected_sources": ["fastapi/background.py"],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does FastAPI serialize the return value of a path operation against the response_model?",
        "expected_answer": "fastapi/routing.py — serialize_response uses the response field to validate then jsonable_encoder to serialize",
        "expected_sources": ["fastapi/routing.py", "fastapi/encoders.py"],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does OAuth2PasswordBearer respond when no Authorization header is present?",
        "expected_answer": "fastapi/security/oauth2.py — raises 401 with WWW-Authenticate: Bearer unless auto_error is False",
        "expected_sources": ["fastapi/security/oauth2.py"],
        "source_type": "code",
        "category": "explanation",
    },

    # ── doc queries (source_type=doc, pinned to English docs) ──
    {
        "query": "How do I install FastAPI with its standard optional dependencies?",
        "expected_answer": 'pip install "fastapi[standard]" inside a virtual environment',
        "expected_sources": ["docs/en/docs/tutorial/index.md"],
        "source_type": "doc",
        "category": "doc",
    },
    {
        "query": "How do I run the development server with the FastAPI CLI?",
        "expected_answer": "fastapi dev main.py — serves on 127.0.0.1:8000 with auto-reload",
        "expected_sources": ["docs/en/docs/fastapi-cli.md", "docs/en/docs/tutorial/index.md"],
        "source_type": "doc",
        "category": "doc",
    },
    {
        "query": "What is the recommended way to deploy a FastAPI app with Docker?",
        "expected_answer": "docs/en/docs/deployment/docker.md — python:3.x slim base image, install requirements, run fastapi run / uvicorn",
        "expected_sources": ["docs/en/docs/deployment/docker.md"],
        "source_type": "doc",
        "category": "doc",
    },
    {
        "query": "How do I set up a development environment to contribute to FastAPI?",
        "expected_answer": "docs/en/docs/contributing.md — clone, create a venv, pip install -e with requirements, run scripts/test.sh",
        "expected_sources": ["docs/en/docs/contributing.md"],
        "source_type": "doc",
        "category": "doc",
    },
    {
        "query": "How should I structure a bigger application with multiple routers across files?",
        "expected_answer": "docs/en/docs/tutorial/bigger-applications.md — app package with routers/ submodules combined via include_router",
        "expected_sources": ["docs/en/docs/tutorial/bigger-applications.md"],
        "source_type": "doc",
        "category": "doc",
    },

    # ── negative: repo genuinely lacks these ──────────
    {
        "query": "Where is the Redis caching layer implemented?",
        "expected_answer": "Not in this repo — the system must say it cannot answer from the sources",
        "expected_sources": [],
        "source_type": "code",
        "category": "negative",
    },
    {
        "query": "How does FastAPI integrate with Kafka for event streaming?",
        "expected_answer": "Not in this repo — the system must say it cannot answer from the sources",
        "expected_sources": [],
        "source_type": "code",
        "category": "negative",
    },
    {
        "query": "Where is the billing and payments module?",
        "expected_answer": "Not in this repo — the system must say it cannot answer from the sources",
        "expected_sources": [],
        "source_type": "code",
        "category": "negative",
    },

    # ── boundary: straddle code and docs (hard-filter stress) ──
    {
        "query": "How does `Depends` work and how do the docs recommend using it?",
        "expected_answer": "fastapi/param_functions.py defines Depends; docs/en/docs/tutorial/dependencies/index.md shows usage",
        "expected_sources": ["fastapi/param_functions.py", "docs/en/docs/tutorial/dependencies/index.md"],
        "source_type": "code",
        "category": "boundary",
    },
    {
        "query": "What does OAuth2PasswordBearer do and how is it used in the security tutorial?",
        "expected_answer": "fastapi/security/oauth2.py defines it; docs/en/docs/tutorial/security/first-steps.md shows the tutorial flow",
        "expected_sources": ["fastapi/security/oauth2.py", "docs/en/docs/tutorial/security/first-steps.md"],
        "source_type": "code",
        "category": "boundary",
    },
]


# ── Retrieval-only scoring ────────────────────────────
def evaluate_retrieval(repo_id: str, top_k: int = 5) -> dict:
    """Score hybrid retrieval + rerank against the golden set. No LLM calls."""
    results = []
    for item in GOLDEN_SET:
        if item["category"] == "negative":
            continue  # needs answer-level eval (faithfulness), not retrieval

        st = item["source_type"]
        # Same defaulting as answer_query: code answers from library source
        pt = "library" if st == "code" else None
        # build_retriever is itself cached (per repo_id/source_type/path_type),
        # so calling it per golden-set item just reuses the same retriever
        # after the first item of each (st, pt) combo.
        retriever = build_retriever(repo_id, source_type=st, path_type=pt)

        raw = retriever.invoke(item["query"])
        candidates = [{"content": d.page_content, "metadata": d.metadata} for d in raw]
        top = rerank(item["query"], candidates, top_k=top_k)
        retrieved = [norm_path(c["metadata"]["rel_path"]) for c in top]

        expected = [norm_path(p) for p in item["expected_sources"]]
        first_hit = next(
            (i for i, path in enumerate(retrieved) if path in expected), None
        )
        translation_miss = first_hit is None and any(
            is_translation_miss(r, e) for r in retrieved for e in expected
        )

        results.append({
            "query": item["query"],
            "category": item["category"],
            "source_type": st,
            "hit": first_hit is not None,
            "reciprocal_rank": 1 / (first_hit + 1) if first_hit is not None else 0.0,
            "translation_miss": translation_miss,
            "retrieved": retrieved,
            "expected": expected,
        })

    scored = [r for r in results if r["category"] != "boundary"]
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["hit"])

    return {
        "hit_rate": sum(r["hit"] for r in scored) / len(scored),
        "mrr": sum(r["reciprocal_rank"] for r in scored) / len(scored),
        "by_category": {
            cat: sum(hits) / len(hits) for cat, hits in by_category.items()
        },
        "results": results,
    }


# ── RAGAS answer-quality scoring ──────────────────────
def evaluate_answers(repo_id: str, limit: int | None = None) -> dict:
    """Score generated answers with RAGAS: faithfulness, answer relevancy,
    context recall. Every item costs a real LLM call to generate the answer,
    plus several more internally for RAGAS's own judge model — pass `limit`
    to bound cost while iterating.

    ground_truth is the golden set's expected_answer (a short, hand-verified
    reference), and contexts is the actual retrieved chunk text (not file
    paths — RAGAS needs real content to judge faithfulness/recall against).
    """
    items = GOLDEN_SET[:limit] if limit else GOLDEN_SET

    rows = []
    categories = []
    for item in items:
        result = answer_query(
            item["query"], repo_id,
            source_type=item["source_type"],
            include_answer=True,
            include_context=True,
        )
        rows.append({
            "question": item["query"],
            "answer": result["answer"],
            "contexts": result["context_chunks"],
            "ground_truth": item["expected_answer"],
        })
        categories.append(item["category"])

    dataset = Dataset.from_list(rows)
    ragas_result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    df = ragas_result.to_pandas()
    df["category"] = categories

    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }

    by_category = {}
    for cat in sorted(set(categories)):
        sub = df[df["category"] == cat]
        by_category[cat] = {
            "faithfulness": float(sub["faithfulness"].mean()),
            "answer_relevancy": float(sub["answer_relevancy"].mean()),
            "context_recall": float(sub["context_recall"].mean()),
        }

    return {
        "scores": scores,
        "by_category": by_category,
        "results": df.to_dict(orient="records"),
    }
