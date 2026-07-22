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

import os
import time

from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.callbacks import get_openai_callback
from ragas.run_config import RunConfig
from datasets import Dataset

from app.retrieval import build_retriever, rerank, answer_query
from app.scoring import norm_path, is_translation_miss, function_hit, ANSWER_QUALITY_THRESHOLDS, check_answer_gate  # noqa: F401

# Candidate judge models under evaluation -- see
# scripts/smoke_tests/judge_calibration.py for the calibration process behind
# this choice, and notes/eval-harness-and-ragas.md / codelens.md's Aspect 4
# addendum for why an explicit judge model matters at all (RAGAS silently
# defaults to gpt-3.5-turbo if `llm=` is never passed, which measured 86% on
# judge_calibration.py's cases vs gpt-4o's 100%, and was observed failing to
# parse RAGAS's own structured-output format on real multi-sentence answers).
#
# Each entry: model name, optional base_url (None = OpenAI's own endpoint),
# and the env var holding the API key for that endpoint. qwen3-8b is not
# OpenAI-hosted, so it's routed through OpenRouter's OpenAI-compatible API
# instead -- confirmed the model slug is real via OpenRouter's own /models
# endpoint before wiring it in (https://openrouter.ai/api/v1/models).
# gemma-4-31b is routed through ASU Research Computing's OpenAI-compatible
# gateway (https://openai.rc.asu.edu/v1) instead of OpenRouter -- same
# underlying model (google/gemma-4-31b-it on OpenRouter, gemma4-31b-it on
# ASU's endpoint -- note the different exact model-name string per
# provider), but ASU-subsidized rather than a per-token OpenRouter charge.
# Either backend avoids self-hosting a 31B model locally (~17-20GB VRAM
# even 4-bit quantized) and both are reachable from CI (no GPU needed,
# unlike local self-hosting).
JUDGE_MODELS = {
    "gpt-4o": {"model": "gpt-4o", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "gpt-5.2": {"model": "gpt-5.2", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "qwen3-8b": {"model": "qwen/qwen3-8b", "base_url": "https://openrouter.ai/api/v1",
                 "api_key_env": "OPENROUTER_API_KEY"},
    "gemma-4-31b": {"model": "gemma4-31b-it", "base_url": "https://openai.rc.asu.edu/v1",
                     "api_key_env": "ASU_RC_API_KEY"},
    # Same model as "gemma-4-31b" above, hosted via OpenRouter instead of
    # ASU RC -- added because ASU RC's endpoint (openai.rc.asu.edu)
    # resolves to private RFC1918 addresses (confirmed via direct curl:
    # 10.139.126.22x), reachable only from ASU's own network/VPN.
    # GitHub-hosted CI runners have no route there, so any real CI run
    # using "gemma-4-31b" fails with a ConnectTimeout every time,
    # deterministically -- this is the entry CI actually uses (see
    # .github/workflows/ci.yml). "gemma-4-31b" stays available for local
    # use, where ASU RC works fine on the VPN.
    "gemma-4-31b-openrouter": {"model": "google/gemma-4-31b-it", "base_url": "https://openrouter.ai/api/v1",
                               "api_key_env": "OPENROUTER_API_KEY"},
    # Free-tier variant on OpenRouter -- zero cost, but confirmed live to be
    # unreliable: a direct curl test returned 429 "temporarily rate-limited
    # upstream" on first try, same as the free generator variant above.
    # Registered as an available option, not a recommended default.
    "gemma-4-31b-openrouter-free": {"model": "google/gemma-4-31b-it:free", "base_url": "https://openrouter.ai/api/v1",
                                    "api_key_env": "OPENROUTER_API_KEY"},
}
# Back to gpt-4o now that the OpenAI account has real credit again ($4.62
# as of this change) -- gpt-4o is the original, most-calibrated judge
# (100% on judge_calibration.py, and the reference the ANSWER_QUALITY_
# THRESHOLDS values in app/scoring.py were calibrated against), and unlike
# ASU RC it's normal public infrastructure -- reachable from CI directly,
# not just from a VPN-connected machine. gemma-4-31b/gemma-4-31b-openrouter
# stay registered as free fallbacks if credit runs out again.
DEFAULT_JUDGE = "gpt-4o"

# langchain_community's get_openai_callback() tracks token counts
# accurately (verified live: 12 prompt + 1 completion tokens for a
# trivial "reply with OK" probe, matching expectations) but its own
# total_cost came back 0.0 despite "gpt-4o" genuinely being present in
# its internal MODEL_COST_PER_1K_TOKENS table -- a real gap in that
# utility for chat-model responses in the installed langchain version,
# not a config error here. Don't trust its cost field; compute cost
# manually from the (verified-accurate) token counts instead, same
# pattern as app/retrieval.py's GENERATOR_PRICING. Only entries actually
# checked against a real pricing page -- not filled in for every
# JUDGE_MODELS key.
JUDGE_PRICING = {
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
}


def build_judge(judge_name: str) -> LangchainLLMWrapper:
    if judge_name not in JUDGE_MODELS:
        raise ValueError(f"Unknown judge '{judge_name}'. Choices: {list(JUDGE_MODELS)}")
    cfg = JUDGE_MODELS[judge_name]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Judge '{judge_name}' needs {cfg['api_key_env']} set in the environment "
            f"(.env locally, or a repo secret in CI) -- not currently set."
        )
    # max_tokens capped explicitly, same as app/retrieval.py's call_llm --
    # without it, ChatOpenAI lets the request default to the model's full
    # context window (gemma-4-31b: 262K), and OpenRouter tries to reserve
    # that many completion tokens up front. Confirmed as a real failure via
    # a live CI run: a 402 "requested up to 40960 tokens, can only afford
    # 2954" even though RAGAS's actual judge task (classify each atomic
    # claim as supported/unsupported, or generate a short reverse-engineered
    # question for answer_relevancy) only ever produces a short, bounded
    # response -- the huge token reservation was never actually needed.
    kwargs = {"model": cfg["model"], "timeout": 120, "api_key": api_key, "max_tokens": 1024}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


# answer_relevancy needs its own embedding model, separate from the judge
# LLM -- ragas_evaluate()'s embeddings= param also silently defaults to
# OpenAI if never passed, the same silent-default trap the judge llm= had.
# Previously routed through ASU RC's qwen3-vl-embedding-8b -- moved to
# reusing app.retrieval's own local HuggingFace embedding model instead,
# after ASU RC's endpoint (openai.rc.asu.edu) was confirmed to resolve to
# private RFC1918 addresses (10.139.126.22x), unreachable from GitHub-hosted
# CI runners. OpenRouter (the fix used for the generator and judge above)
# doesn't offer embeddings at all -- confirmed directly, its /embeddings
# endpoint returns a plain 404 and no embedding models appear in its
# catalog -- so there's no equivalent swap available there. Reusing the
# local model instead of finding a third external provider is not a
# workaround: the retrieval pipeline already downloads and runs this exact
# model locally (confirmed working in CI via the retrieval-only eval gate),
# so this removes an external network dependency this step never actually
# needed, rather than trading one remote provider for another.
def build_answer_relevancy_embeddings() -> LangchainEmbeddingsWrapper:
    from app.retrieval import embedding_fn
    return LangchainEmbeddingsWrapper(embedding_fn)

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
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "APIRouter", "func_name": "add_api_route"},
            {"rel_path": "fastapi/applications.py", "class_name": "FastAPI", "func_name": "add_api_route"},
        ],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where is OAuth2 password bearer authentication implemented?",
        "expected_answer": "fastapi/security/oauth2.py — OAuth2PasswordBearer extracts the bearer token from the Authorization header",
        "expected_sources": ["fastapi/security/oauth2.py"],
        "expected_functions": [
            {"rel_path": "fastapi/security/oauth2.py", "class_name": "OAuth2PasswordBearer", "func_name": "__call__"},
        ],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where are WebSocket routes registered?",
        "expected_answer": "fastapi/routing.py — add_api_websocket_route creates an APIWebSocketRoute (also exposed on FastAPI in applications.py)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "APIRouter", "func_name": "add_api_websocket_route"},
            {"rel_path": "fastapi/applications.py", "class_name": "FastAPI", "func_name": "add_api_websocket_route"},
        ],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where are validation errors converted into HTTP responses?",
        "expected_answer": "fastapi/exception_handlers.py — request_validation_exception_handler returns a 422 JSONResponse",
        "expected_sources": ["fastapi/exception_handlers.py"],
        "expected_functions": [
            {"rel_path": "fastapi/exception_handlers.py", "class_name": "", "func_name": "request_validation_exception_handler"},
        ],
        "source_type": "code",
        "category": "localization",
    },
    {
        "query": "Where does FastAPI validate HTTP Basic auth credentials?",
        "expected_answer": "fastapi/security/http.py — HTTPBasic parses and decodes the Authorization header",
        "expected_sources": ["fastapi/security/http.py"],
        "expected_functions": [
            {"rel_path": "fastapi/security/http.py", "class_name": "HTTPBasic", "func_name": "__call__"},
        ],
        "source_type": "code",
        "category": "localization",
    },

    # ── code: exact identifiers (BM25 stress tests) ───
    {
        "query": "Where is `add_api_route` defined?",
        "expected_answer": "fastapi/routing.py (APIRouter.add_api_route) and fastapi/applications.py (FastAPI.add_api_route)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "APIRouter", "func_name": "add_api_route"},
            {"rel_path": "fastapi/applications.py", "class_name": "FastAPI", "func_name": "add_api_route"},
        ],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `solve_dependencies` implemented?",
        "expected_answer": "fastapi/dependencies/utils.py — async solve_dependencies resolves the dependency tree for a request",
        "expected_sources": ["fastapi/dependencies/utils.py"],
        "expected_functions": [
            {"rel_path": "fastapi/dependencies/utils.py", "class_name": "", "func_name": "solve_dependencies"},
        ],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "What does `jsonable_encoder` do?",
        "expected_answer": "fastapi/encoders.py — recursively converts objects (models, dataclasses, datetimes) into JSON-compatible types",
        "expected_sources": ["fastapi/encoders.py"],
        "expected_functions": [
            {"rel_path": "fastapi/encoders.py", "class_name": "", "func_name": "jsonable_encoder"},
        ],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `serialize_response` implemented?",
        "expected_answer": "fastapi/routing.py — serialize_response validates and serializes the return value against the response model",
        "expected_sources": ["fastapi/routing.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "", "func_name": "serialize_response"},
        ],
        "source_type": "code",
        "category": "identifier",
    },
    {
        "query": "Where is `include_router` defined?",
        "expected_answer": "fastapi/routing.py (APIRouter.include_router) and fastapi/applications.py (FastAPI.include_router)",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "APIRouter", "func_name": "include_router"},
            {"rel_path": "fastapi/applications.py", "class_name": "FastAPI", "func_name": "include_router"},
        ],
        "source_type": "code",
        "category": "identifier",
    },

    # ── code: explanation ─────────────────────────────
    {
        "query": "How does dependency injection resolve nested dependencies?",
        "expected_answer": "fastapi/dependencies/utils.py — solve_dependencies walks the dependant tree, awaiting sub-dependencies and caching results",
        "expected_sources": ["fastapi/dependencies/utils.py"],
        "expected_functions": [
            {"rel_path": "fastapi/dependencies/utils.py", "class_name": "", "func_name": "solve_dependencies"},
        ],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does `Depends` declare a dependency for a path operation parameter?",
        "expected_answer": "fastapi/param_functions.py — Depends returns a params.Depends marker consumed during dependant analysis",
        "expected_sources": ["fastapi/param_functions.py"],
        "expected_functions": [
            {"rel_path": "fastapi/param_functions.py", "class_name": "", "func_name": "Depends"},
        ],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How are background tasks executed after a response is sent?",
        "expected_answer": "fastapi/background.py — BackgroundTasks collects tasks that Starlette runs after the response",
        "expected_sources": ["fastapi/background.py"],
        "expected_functions": [
            {"rel_path": "fastapi/background.py", "class_name": "BackgroundTasks", "func_name": "add_task"},
        ],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does FastAPI serialize the return value of a path operation against the response_model?",
        "expected_answer": "fastapi/routing.py — serialize_response uses the response field to validate then jsonable_encoder to serialize",
        "expected_sources": ["fastapi/routing.py", "fastapi/encoders.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "", "func_name": "serialize_response"},
            {"rel_path": "fastapi/encoders.py", "class_name": "", "func_name": "jsonable_encoder"},
        ],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How does OAuth2PasswordBearer respond when no Authorization header is present?",
        "expected_answer": "fastapi/security/oauth2.py — raises 401 with WWW-Authenticate: Bearer unless auto_error is False",
        "expected_sources": ["fastapi/security/oauth2.py"],
        "expected_functions": [
            {"rel_path": "fastapi/security/oauth2.py", "class_name": "OAuth2PasswordBearer", "func_name": "__call__"},
        ],
        "source_type": "code",
        "category": "explanation",
    },
    {
        "query": "How are path operations configured?",
        # Verified directly against source, not copied from a generated
        # answer -- confirmed api_route() is defined in both routing.py
        # (APIRouter) and applications.py (FastAPI), and confirmed the
        # HTTP-verb decorators (get/post/put/delete/patch/options/head)
        # each internally call `self.api_route(...)`, passing the
        # appropriate `methods` list -- 9 call sites of `self.api_route(`
        # found in routing.py alone.
        "expected_answer": "Decorator methods (@app.get, @app.post, @app.put, "
                           "@app.delete, @app.patch, @app.options, @app.head) on both "
                           "APIRouter (fastapi/routing.py) and FastAPI (fastapi/applications.py) "
                           "each internally call api_route(), passing the appropriate HTTP "
                           "methods list -- api_route is the actual shared registration logic",
        "expected_sources": ["fastapi/routing.py", "fastapi/applications.py"],
        "expected_functions": [
            {"rel_path": "fastapi/routing.py", "class_name": "APIRouter", "func_name": "api_route"},
            {"rel_path": "fastapi/applications.py", "class_name": "FastAPI", "func_name": "api_route"},
        ],
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
        # None (not "code") -- this item's own ground truth spans both code
        # and docs, so pinning source_type here would filter out the docs
        # half of the answer before retrieval even runs, guaranteeing a low
        # context_recall/faithfulness regardless of system quality. Search
        # everything, matching what "boundary" is meant to stress-test.
        "source_type": None,
        "category": "boundary",
    },
    {
        "query": "What does OAuth2PasswordBearer do and how is it used in the security tutorial?",
        "expected_answer": "fastapi/security/oauth2.py defines it; docs/en/docs/tutorial/security/first-steps.md shows the tutorial flow",
        "expected_sources": ["fastapi/security/oauth2.py", "docs/en/docs/tutorial/security/first-steps.md"],
        "source_type": None,
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

        # Function-level: did retrieval get the *specific function*, not just
        # the right file? Only scored for items with hand-verified
        # expected_functions (localization/identifier/explanation) -- see
        # codelens.md's Aspect 4 addendum for why file-level hit_rate alone
        # missed a real case (add_api_route never retrieved despite 4/5
        # chunks being in the right file).
        func_hit = None
        if "expected_functions" in item:
            retrieved_funcs = [
                {
                    "rel_path": c["metadata"]["rel_path"],
                    "class_name": c["metadata"].get("class_name", ""),
                    "func_name": c["metadata"].get("func_name", ""),
                }
                for c in top
            ]
            func_hit = function_hit(retrieved_funcs, item["expected_functions"])

        results.append({
            "query": item["query"],
            "category": item["category"],
            "source_type": st,
            "hit": first_hit is not None,
            "reciprocal_rank": 1 / (first_hit + 1) if first_hit is not None else 0.0,
            "translation_miss": translation_miss,
            "function_hit": func_hit,
            "retrieved": retrieved,
            "expected": expected,
        })

    scored = [r for r in results if r["category"] != "boundary"]
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["hit"])

    func_scored = [r for r in results if r["function_hit"] is not None]
    func_by_category = {}
    for r in func_scored:
        func_by_category.setdefault(r["category"], []).append(r["function_hit"])

    return {
        "hit_rate": sum(r["hit"] for r in scored) / len(scored),
        "mrr": sum(r["reciprocal_rank"] for r in scored) / len(scored),
        "by_category": {
            cat: sum(hits) / len(hits) for cat, hits in by_category.items()
        },
        # Function-level hit rate: stricter than hit_rate, only computed over
        # items with expected_functions ground truth (a subset of `scored`).
        "function_hit_rate": (
            sum(func_scored[i]["function_hit"] for i in range(len(func_scored))) / len(func_scored)
            if func_scored else None
        ),
        "function_hit_rate_by_category": {
            cat: sum(hits) / len(hits) for cat, hits in func_by_category.items()
        },
        "results": results,
    }


# ── RAGAS answer-quality scoring ──────────────────────
def evaluate_answers(repo_id: str, limit: int | None = None, judge: str = DEFAULT_JUDGE) -> dict:
    """Score generated answers with RAGAS: faithfulness, answer relevancy.
    (context_recall was measured for a while, then dropped entirely -- see
    app/scoring.py's comment for why.) Every item costs a real LLM call to
    generate the answer, plus several more internally for RAGAS's own
    judge model — pass `limit` to bound cost while iterating.

    `judge` selects which model grades the answers -- see JUDGE_MODELS.
    This is a separate choice from the model that generates the answers,
    controlled independently via app/retrieval.py's GENERATOR_MODEL env var
    (defaults to an ASU RC model as of this writing -- see that module's
    GENERATOR_MODELS registry for why, and codelens.md's Aspect 4 addendum
    for the OpenAI credit situation that motivated it). Historical
    faithfulness/context_recall numbers documented in codelens.md were
    measured against gpt-4o-generated answers specifically -- comparing
    judges here is about calibrating the grader, not the generator, but
    changing the generator does mean future runs aren't perfectly
    apples-to-apples with those earlier numbers unless GENERATOR_MODEL is
    set back to gpt-4o.

    ground_truth is the golden set's expected_answer (a short, hand-verified
    reference, unused by either metric now that context_recall is gone --
    kept in the dataset for whichever future metric might need it), and
    contexts is the actual retrieved chunk text (not file paths — RAGAS
    needs real content to judge faithfulness/relevancy against).
    """
    items = GOLDEN_SET[:limit] if limit else GOLDEN_SET

    rows = []
    categories = []
    generator_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    # Timed as two separate phases, not one total -- they have genuinely
    # different bottlenecks and failure sources (generation: per-item
    # provider hiccups/rate limits; judging: RAGAS's own serialized judge
    # calls, see the max_workers=1 comment below) -- collapsing them into
    # one number would hide which phase is actually slow on a given run.
    generation_start = time.monotonic()
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
        usage = result.get("token_usage")
        if usage:
            generator_usage["prompt_tokens"] += usage["prompt_tokens"]
            generator_usage["completion_tokens"] += usage["completion_tokens"]
            generator_usage["total_tokens"] += usage["total_tokens"]
            if usage["estimated_cost_usd"] is None:
                generator_usage["estimated_cost_usd"] = None
            elif generator_usage["estimated_cost_usd"] is not None:
                generator_usage["estimated_cost_usd"] += usage["estimated_cost_usd"]

    generation_seconds = time.monotonic() - generation_start

    dataset = Dataset.from_list(rows)
    judge_llm = build_judge(judge)
    judge_embeddings = build_answer_relevancy_embeddings()
    # Default max_workers=16 fires that many concurrent judge calls at once --
    # a real run hit a hard 429 (gpt-4o TPM cap for this org is 30,000 --
    # a tier-1 default) even at max_workers=3: 3 concurrent judge calls
    # (faithfulness alone makes 2 calls/item) plus tenacity's own retries
    # re-saturate the same per-minute ceiling faster than it drains, so a
    # long enough run always eventually catches a request landing on an
    # already-near-full window. Fully serializing (max_workers=1) means at
    # most one in-flight request at a time, so total tokens/min tracks the
    # judge's own call rate instead of being multiplied by concurrency --
    # slower wall-clock, but actually bounded by the real TPM limit.
    # context_recall dropped entirely (was already report-only, never
    # gated) -- structurally unwinnable for the "negative" category (its
    # ground truth is an absence claim no retrieved text can ever support)
    # and noisy run-to-run on the rest (0.0-0.4 swings observed on
    # identical code across real runs). Was ~25% of judge call volume for
    # a signal that couldn't fail the build and wasn't being acted on --
    # see app/scoring.py's ANSWER_QUALITY_THRESHOLDS comment for the full
    # history (recalibrated -> demoted to report-only -> dropped here).
    judging_start = time.monotonic()
    with get_openai_callback() as cb:
        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(max_workers=1),
        )
    judging_seconds = time.monotonic() - judging_start

    # cb's own total_cost is NOT used -- verified live it comes back 0.0
    # even for "gpt-4o" (which genuinely is in langchain's pricing table),
    # a real gap in that utility for chat-model responses. Its token
    # counts ARE verified accurate; cost computed manually from those,
    # same pattern as app/retrieval.py's GENERATOR_PRICING.
    judge_pricing = JUDGE_PRICING.get(judge)
    judge_usage = {
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
        "estimated_cost_usd": (
            cb.prompt_tokens / 1_000_000 * judge_pricing["input_per_1m"]
            + cb.completion_tokens / 1_000_000 * judge_pricing["output_per_1m"]
        ) if judge_pricing else None,
    }

    df = ragas_result.to_pandas()
    df["category"] = categories

    # "negative" excluded from the answer_relevancy GATE average only --
    # still fully computed and visible per-category below, just not
    # allowed to sink the pass/fail number. Root-caused, not guessed: read
    # RAGAS's actual instruction prompt for this metric
    # (ragas/metrics/_answer_relevance.py) -- it names "I don't know" as
    # the canonical example of a "noncommittal" answer, and noncommittal
    # answers get their score hard-multiplied by 0
    # (score = cosine_sim.mean() * int(not committal)), independent of
    # how topically correct the answer actually was. "negative" items'
    # ground truth IS "the system must say it cannot answer" -- i.e. the
    # exact answer shape this metric is built to zero out. Confirmed live
    # against a real run: 2 of 3 negative items scored an exact 0.0 (not
    # low-but-nonzero) despite being clean, correct refusals -- a hard
    # classifier artifact, not a real quality signal. faithfulness stays
    # a full mean across all categories -- it showed real, non-trivial
    # variance on these same items (1.0, 0.571, 0.875), unlike
    # answer_relevancy's proven hard-zero, so there's no equivalent case
    # to exclude it there too.
    non_negative = df["category"] != "negative"
    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df.loc[non_negative, "answer_relevancy"].mean()),
    }

    by_category = {}
    for cat in sorted(set(categories)):
        sub = df[df["category"] == cat]
        by_category[cat] = {
            "faithfulness": float(sub["faithfulness"].mean()),
            "answer_relevancy": float(sub["answer_relevancy"].mean()),
        }

    return {
        "judge": judge,
        "scores": scores,
        "by_category": by_category,
        "generator_usage": generator_usage,
        "judge_usage": judge_usage,
        "timing": {
            "generation_seconds": round(generation_seconds, 1),
            "judging_seconds": round(judging_seconds, 1),
            "total_seconds": round(generation_seconds + judging_seconds, 1),
        },
        "results": df.to_dict(orient="records"),
    }
