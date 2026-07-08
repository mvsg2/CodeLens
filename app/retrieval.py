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
from openai import OpenAI

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

# ── OpenAI client ─────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)


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
def build_retriever(repo_id: str, source_type: str = "code",
                    path_type: str | None = None) -> EnsembleRetriever:
    documents = load_documents(repo_id)

    # Semantic retriever, filtered at the Chroma level
    conditions = [{"source_type": source_type}]
    if path_type:
        conditions.append({"path_type": path_type})
    chroma_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    vectorstore = Chroma(
        collection_name="codebase",
        persist_directory=str(CHROMA_DIR / repo_id),
        embedding_function=embedding_fn
    )
    semantic = vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": chroma_filter}
    )

    # Keyword retriever over the same subset
    filtered = [
        d for d in documents
        if d.metadata.get("source_type") == source_type
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
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )
    top = []
    for score, chunk in ranked[:top_k]:
        # Cross-encoder relevance score, previously computed then discarded —
        # kept here so callers can judge confidence, not just get an
        # unqualified top-5 regardless of whether any of them are any good.
        chunk["rerank_score"] = float(score)
        top.append(chunk)
    return top


# ── Build prompt ──────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        line = meta.get("start_line", "?")
        func = meta.get("func_name", "")
        ref = f"{meta['rel_path']}:{line}"
        if func:
            ref += f" ({func})"
        block = f"[SOURCE {i+1}: {ref}]\n```{meta.get('extension','').lstrip('.')}\n{chunk['content']}\n```"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

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
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content


# ── Main answer function ──────────────────────────────
def answer_query(query: str, repo_id: str, source_type: str = "code",
                 path_type: str | None = "auto", include_answer: bool = True,
                 include_context: bool = False) -> dict:
    # "auto": code queries answer from library source; doc queries search all docs.
    # Pass None to disable the path filter, or e.g. "examples" to target docs_src/.
    if path_type == "auto":
        path_type = "library" if source_type == "code" else None

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
    if include_answer:
        print("Calling LLM...")
        prompt = build_prompt(query, top_chunks)
        answer = call_llm(prompt)
    else:
        print("Skipping LLM call (--no-llm)")

    result = {
        "answer": answer,
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
        # Raw chunk text, not just file paths — RAGAS needs actual content to
        # judge faithfulness/context recall against, not a filename string.
        result["context_chunks"] = [c["content"] for c in top_chunks]
    return result