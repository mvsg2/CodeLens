import os
import time
from functools import lru_cache
from pathlib import Path
import chromadb
import torch
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import OpenAI, RateLimitError

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME

CHROMA_DIR = Path("data/chroma")

# ── Embedding function (same model as pipeline) ───────
embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={
        "trust_remote_code": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    encode_kwargs={"normalize_embeddings": True}
)

# ── Reranker ──────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Answer-generation model ───────────────────────────
# Swappable via GENERATOR_MODEL env var (defaults to the ASU RC candidate
# below) -- temporary measure while the OpenAI account's credit balance is
# unresolved (see notes/ and codelens.md's Aspect 4 addendum for the real
# insufficient_quota investigation). Kept as a small registry, same shape
# as app/eval.py's JUDGE_MODELS, so switching back to gpt-4o/gpt-5.2 once
# credit is available is a one-line env var change, not a code change --
# this is meant to be reversible, not a permanent architecture decision.
# qwen3-coder-30b-a3b-instruct chosen deliberately over ASU's larger
# general-purpose models: coder-specialized (better domain fit for a
# code-Q&A system), MoE with ~3B active params (cheaper/faster to serve
# than a same-size dense model), 131K context (far more than the ~5
# retrieved chunks per query actually need).
GENERATOR_MODELS = {
    "gpt-4o": {"model": "gpt-4o", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "gpt-5.2": {"model": "gpt-5.2", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "qwen3-coder-30b": {"model": "qwen3-coder-30b-a3b-instruct",
                        "base_url": "https://openai.rc.asu.edu/v1", "api_key_env": "ASU_RC_API_KEY"},
    # 0.8B dense model -- smallest ASU RC candidate, tried as a
    # lower-latency alternative to the 30B MoE coder model above after a
    # /query timeout (root cause turned out to be a dropped VPN connection,
    # not model speed). Not used as default: its responses come back with
    # message.content == null (text lands in message.reasoning_content
    # instead), which call_llm() doesn't read -- would silently return
    # empty answers without a code change. Kept registered for reference.
    "qwen35-0p8b": {"model": "qwen35-0p8b",
                    "base_url": "https://openai.rc.asu.edu/v1", "api_key_env": "ASU_RC_API_KEY"},
    # Tried briefly as the default (fast, no reasoning_content quirk), then
    # reverted: real /query test showed reproducible garbled output (stray
    # non-English tokens mid-answer across two separate runs) -- a genuine
    # generation-quality defect, not a fluke. Kept registered for reference,
    # not the default.
    "gemma4-e2b-it": {"model": "gemma4-e2b-it",
                      "base_url": "https://openai.rc.asu.edu/v1", "api_key_env": "ASU_RC_API_KEY"},
    # Same underlying model as "qwen3-coder-30b" above, hosted via
    # OpenRouter instead of ASU RC -- added specifically because ASU RC's
    # endpoint (openai.rc.asu.edu) resolves to private RFC1918 addresses
    # (confirmed via direct curl: 10.139.126.22x), reachable only from
    # ASU's own network/VPN. GitHub-hosted CI runners have no route there
    # at all, so any eval step depending on the ASU RC entry fails with a
    # ConnectTimeout every single run, deterministically -- not something
    # retries or a longer timeout can fix. This entry is what CI actually
    # uses (see .github/workflows/ci.yml's GENERATOR_MODEL override);
    # "qwen3-coder-30b" above stays the local-dev default since ASU RC
    # works fine from a machine that's actually on the VPN.
    "qwen3-coder-30b-openrouter": {"model": "qwen/qwen3-coder-30b-a3b-instruct",
                                   "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    # Free-tier variant of the larger qwen3-coder model on OpenRouter --
    # zero cost, no credit balance needed, but confirmed live (not
    # hypothetical) to be unreliable: a direct curl test returned 429
    # "temporarily rate-limited upstream" on first try. Registered as an
    # available option, not a recommended default -- something CI depends
    # on to pass reproducibly shouldn't lean on a free tier already
    # observed rate-limiting under light, one-off testing.
    "qwen3-coder-openrouter-free": {"model": "qwen/qwen3-coder:free",
                                    "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    # Hosted directly by DeepSeek (not via OpenRouter) -- own paid account,
    # no OpenRouter/ASU RC dependency at all. "-flash" is the fast,
    # non-thinking mode (vs. "-pro"'s thinking mode) -- the right fit here
    # since generator calls are capped at max_tokens=4096 for concise RAG
    # answers, not long reasoning chains. Different vendor family from the
    # gpt-4o judge, so no generator/judge circularity concern.
    "deepseek-v4-flash": {"model": "deepseek-v4-flash",
                          "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
}
GENERATOR_MODEL_NAME = os.environ.get("GENERATOR_MODEL", "qwen3-coder-30b")

# Only entries actually verified against a real pricing page this session
# -- deliberately not filled in for every GENERATOR_MODELS key. Guessing a
# number here would be worse than reporting "unknown": a wrong estimate
# looks authoritative and nobody would think to double-check it. Dollars
# per 1M tokens.
GENERATOR_PRICING = {
    "deepseek-v4-flash": {"input_per_1m": 0.14, "output_per_1m": 0.28},  # cache-miss pricing
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
}


def _usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    pricing = GENERATOR_PRICING.get(GENERATOR_MODEL_NAME)
    cost = None
    if pricing:
        cost = (prompt_tokens / 1_000_000 * pricing["input_per_1m"]
                + completion_tokens / 1_000_000 * pricing["output_per_1m"])
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": cost,
    }


def _build_generator_client() -> tuple[OpenAI, str]:
    if GENERATOR_MODEL_NAME not in GENERATOR_MODELS:
        raise ValueError(f"Unknown GENERATOR_MODEL '{GENERATOR_MODEL_NAME}'. Choices: {list(GENERATOR_MODELS)}")
    cfg = GENERATOR_MODELS[GENERATOR_MODEL_NAME]
    api_key = os.environ.get(cfg["api_key_env"]) if cfg["api_key_env"] != "OPENAI_API_KEY" else OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            f"Generator '{GENERATOR_MODEL_NAME}' needs {cfg['api_key_env']} set in the environment."
        )
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg["model"]


client, GENERATOR_MODEL = _build_generator_client()


# ── Load documents from Chroma ────────────────────────
# Cached per repo_id: this reads every chunk in the collection (~20k+ for
# fastapi/fastapi), so re-running it on every single query is wasteful —
# especially over a Docker bind-mounted volume, where repeated small reads
# are noticeably slower than on a native filesystem. The cache only needs
# invalidating when a repo is re-indexed, which happens in a fresh process
# anyway (a new pipeline run / container restart), so a plain in-memory
# cache is enough — no manual invalidation needed.
@lru_cache(maxsize=8)
def load_documents(repo_id: str) -> list[Document]:
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR / repo_id)
    )
    collection = chroma_client.get_collection("codebase")
    results = collection.get(include=["documents", "metadatas"])

    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


# ── Build hybrid retriever ────────────────────────────
# Cached per (repo_id, source_type, path_type): building the BM25 index is
# real work (tokenizing every matching chunk), and repeating it per request
# is the other half of the redundant work removed by caching load_documents
# above. Takes only hashable args (no documents list) so lru_cache can key
# on them directly; it fetches documents itself via the cached function.
@lru_cache(maxsize=32)
def build_retriever(repo_id: str, source_type: str | None = None,
                    path_type: str | None = None) -> EnsembleRetriever:
    documents = load_documents(repo_id)

    # Semantic retriever, filtered at the Chroma level. Both filters are
    # optional -- None means "don't restrict on this dimension," so the
    # default (both None) searches the whole collection: code and docs,
    # library and tests and examples, all together, ranked purely on
    # relevance rather than pre-excluded by a bucket the caller didn't ask
    # to exclude.
    conditions = []
    if source_type:
        conditions.append({"source_type": source_type})
    if path_type:
        conditions.append({"path_type": path_type})
    if not conditions:
        chroma_filter = None
    elif len(conditions) == 1:
        chroma_filter = conditions[0]
    else:
        chroma_filter = {"$and": conditions}

    vectorstore = Chroma(
        collection_name="codebase",
        persist_directory=str(CHROMA_DIR / repo_id),
        embedding_function=embedding_fn
    )
    search_kwargs = {"k": 10}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    semantic = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Keyword retriever over the same subset
    filtered = [
        d for d in documents
        if (not source_type or d.metadata.get("source_type") == source_type)
        and (not path_type or d.metadata.get("path_type") == path_type)
    ]
    keyword = BM25Retriever.from_documents(filtered or documents)
    keyword.k = 10

    # 60% semantic, 40% keyword
    return EnsembleRetriever(
        retrievers=[semantic, keyword],
        weights=[0.6, 0.4]
    )


# ── Rerank ────────────────────────────────────────────
def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def rerank(query: str, chunks: list[dict], top_k: int = 5,
          diversity_lambda: float = 0.7) -> list[dict]:
    """Maximal Marginal Relevance reranking, not a plain top-k-by-score sort.

    Plain top-k lets a cluster of near-duplicate chunks collectively crowd
    out one genuinely distinct, more useful result -- confirmed as a real
    problem once source_type/path_type filtering became optional (see
    codelens.md): searching a repo's whole collection unfiltered surfaced
    several near-identical translated doc pages (same content, different
    language) ranked above the actual relevant code. This isn't specific to
    translation -- the same failure mode applies to versioned docs, vendored
    dependency copies, or repeated boilerplate in a monorepo, none of which
    this function assumes anything about.

    Each candidate is scored as (diversity_lambda * relevance) - ((1 -
    diversity_lambda) * max_similarity_to_already_selected), picking one
    chunk at a time greedily. The similarity term uses the same semantic
    embeddings used for indexing (not literal text overlap), specifically
    because that's what catches near-duplicates whose literal text barely
    overlaps at all -- e.g. the same doc page translated into Turkish,
    English, and Chinese share almost no tokens, but should embed close
    together if the embedding model captures the content's meaning.
    diversity_lambda=0.7 keeps relevance dominant while still penalizing
    near-duplicates; 1.0 would reduce to the old plain top-k-by-score
    behavior, 0.0 would ignore relevance entirely and pick only for spread.
    """
    pairs = [(query, c["content"]) for c in chunks]
    raw_scores = [float(s) for s in reranker.predict(pairs)]

    # Min-max normalize onto [0, 1] -- the cross-encoder's raw scores and
    # cosine similarity live on different scales, so combining them
    # unnormalized would let one term dominate regardless of
    # diversity_lambda instead of expressing the intended trade-off.
    lo, hi = min(raw_scores), max(raw_scores)
    span = hi - lo or 1.0
    norm_scores = [(s - lo) / span for s in raw_scores]

    embeddings = embedding_fn.embed_documents([c["content"] for c in chunks])

    remaining = list(range(len(chunks)))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        def mmr_score(i: int) -> float:
            if not selected:
                return norm_scores[i]
            penalty = max(_cosine_sim(embeddings[i], embeddings[j]) for j in selected)
            return diversity_lambda * norm_scores[i] - (1 - diversity_lambda) * penalty
        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    top = []
    for i in selected:
        # Real cross-encoder relevance score (not the MMR-adjusted value) —
        # callers judging confidence should see the same scale as before,
        # not a number that's already been penalized for diversity.
        chunks[i]["rerank_score"] = raw_scores[i]
        top.append(chunks[i])
    return top


# ── Build prompt ──────────────────────────────────────
def source_blocks(chunks: list[dict]) -> list[str]:
    """[SOURCE N: path:line (func)] + code, per chunk — the exact text the
    LLM is grounded on and instructed to cite from. Used both to build the
    generation prompt and as RAGAS's `contexts` (see answer_query below) so
    the faithfulness judge checks claims against the same material the
    answer was actually grounded on, not a stripped-down version of it.
    Splitting this out fixed a real bug: RAGAS was previously given only
    bare chunk content with no file/line info, so it marked GPT-4o's
    correctly-cited file paths and line numbers as "unsupported" — verified
    by a controlled test showing faithfulness on a real example jump from
    0.333 to 0.667 once RAGAS could see what the answer was actually citing.
    """
    blocks = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        line = meta.get("start_line", "?")
        func = meta.get("func_name", "")
        ref = f"{meta['rel_path']}:{line}"
        if func:
            ref += f" ({func})"
        blocks.append(f"[SOURCE {i+1}: {ref}]\n```{meta.get('extension','').lstrip('.')}\n{chunk['content']}\n```")
    return blocks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(source_blocks(chunks))

    return f"""You are a codebase expert assistant. Answer the developer's question using ONLY the source code provided below.

Rules:
- Cite every claim using [SOURCE N] notation
- Include the exact file path and function name when referencing code
- If the answer cannot be found in the sources, say so explicitly — do not guess or hallucinate
- Be concise and technical

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""


# ── LLM call ─────────────────────────────────────────
def call_llm(prompt: str, max_attempts: int = 5) -> tuple[str, dict]:
    """Returns (answer_text, usage_dict). usage_dict has prompt_tokens/
    completion_tokens/total_tokens (always populated -- every OpenAI-
    compatible response includes these) and estimated_cost_usd (None
    unless GENERATOR_PRICING has a verified entry for the current model).
    """
    # Two distinct transient failure modes handled here, both worth
    # retrying rather than crashing the whole eval run:
    #  1. OpenRouter occasionally returns HTTP 200 with choices=None -- a
    #     hiccup in whichever backend it routed this model to that round
    #     (OpenRouter fans a single model name out across multiple actual
    #     providers). A fresh request usually routes around it.
    #  2. RateLimitError (429) -- expected and routine on free-tier
    #     generators like qwen3-coder-openrouter-free, which share limited
    #     capacity across everyone using it for free. Back off longer here
    #     than case 1 since the limiting window is usually
    #     seconds-to-a-minute, not an instant backend fluke.
    last_response = None
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                # Was 1024 -- confirmed too tight for deepseek-v4-flash's
                # answer style via a real /query test: a genuine answer got
                # cut off mid-word ("...via `get_parameterless_sub_depend
                # ant") instead of finishing, which drags both faithfulness
                # (an incomplete claim can't be verified) and
                # answer_relevancy down. 4096 gives ~4x headroom over what
                # just proved insufficient -- the prompt already asks for
                # "concise and technical" answers, so this is slack for a
                # more verbose model, not an invitation to ramble; cost
                # impact is negligible regardless (DeepSeek: ~$0.001 for a
                # full 4096-token completion).
                max_tokens=4096
            )
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
            continue

        if response.choices:
            return response.choices[0].message.content, _usage_dict(response)
        last_response = response
        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Generator '{GENERATOR_MODEL_NAME}' returned no choices after "
        f"{max_attempts} attempts -- raw response: {last_response!r}"
    )


# ── Main answer function ──────────────────────────────
def answer_query(query: str, repo_id: str, source_type: str | None = None,
                 path_type: str | None = None, include_answer: bool = True,
                 include_context: bool = False) -> dict:
    # Both filters default to None: no restriction, search code + docs,
    # library + tests + examples, all together. Pass an explicit value for
    # either (e.g. source_type="code", path_type="tests") to narrow the
    # search the way the old "code"/"auto" defaults used to narrow it
    # implicitly.
    print(f"Building retriever for: {repo_id} (cached after first call)...")
    retriever = build_retriever(repo_id, source_type=source_type, path_type=path_type)

    print("Retrieving candidates...")
    raw_results = retriever.invoke(query)

    candidates = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in raw_results
    ]

    print(f"Reranking {len(candidates)} candidates...")
    top_chunks = rerank(query, candidates, top_k=5)

    answer = None
    token_usage = None
    if include_answer:
        print("Calling LLM...")
        prompt = build_prompt(query, top_chunks)
        answer, token_usage = call_llm(prompt)
    else:
        print("Skipping LLM call (--no-llm)")

    result = {
        "answer": answer,
        "token_usage": token_usage,
        "sources": [
            {
                "file": c["metadata"]["rel_path"],
                "line": c["metadata"].get("start_line"),
                "function": c["metadata"].get("func_name"),
                "class": c["metadata"].get("class_name"),
                # Cross-encoder relevance score for this (query, chunk) pair.
                # Higher (less negative) = more relevant, but only relative to
                # other candidates for THIS query — not comparable across
                # different queries, so this is a transparency signal, not a
                # calibrated confidence percentage. See notes/ for why a fixed
                # threshold on this score was tried and rejected: a real
                # unanswerable query scored higher than a real good match.
                "relevance_score": c["rerank_score"],
            }
            for c in top_chunks
        ],
        "query": query,
        "repo": repo_id
    }
    if include_context:
        # Same source_blocks() the generation prompt used (file/line headers
        # included), not just bare chunk content — RAGAS's faithfulness judge
        # needs to see what the answer was actually grounded on, or it marks
        # correctly-cited file paths/line numbers as unsupported. See
        # source_blocks()'s docstring for the measured before/after.
        result["context_chunks"] = source_blocks(top_chunks)
    return result