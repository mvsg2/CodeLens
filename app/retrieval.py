import os
from pathlib import Path
import chromadb
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import OpenAI

from app.config import OPENAI_API_KEY

CHROMA_DIR = Path("data/chroma")

# ── Embedding function (same model as pipeline) ───────
embedding_fn = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-code",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# ── Reranker ──────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── OpenAI client ─────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)


# ── Load documents from Chroma ────────────────────────
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
def build_retriever(repo_id: str, documents: list[Document]) -> EnsembleRetriever:
    # Semantic retriever
    vectorstore = Chroma(
        collection_name="codebase",
        persist_directory=str(CHROMA_DIR / repo_id),
        embedding_function=embedding_fn
    )
    semantic = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Keyword retriever
    keyword = BM25Retriever.from_documents(documents)
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
    return [chunk for _, chunk in ranked[:top_k]]


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
def answer_query(query: str, repo_id: str) -> dict:
    print(f"Loading documents for: {repo_id}")
    documents = load_documents(repo_id)

    print(f"Building retriever...")
    retriever = build_retriever(repo_id, documents)

    print(f"Retrieving candidates...")
    raw_results = retriever.invoke(query)

    candidates = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in raw_results
    ]

    print(f"Reranking {len(candidates)} candidates...")
    top_chunks = rerank(query, candidates, top_k=5)

    print(f"Calling LLM...")
    prompt = build_prompt(query, top_chunks)
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "file": c["metadata"]["rel_path"],
                "line": c["metadata"].get("start_line"),
                "function": c["metadata"].get("func_name"),
                "class": c["metadata"].get("class_name")
            }
            for c in top_chunks
        ],
        "query": query,
        "repo": repo_id
    }