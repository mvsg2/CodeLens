# CodeLens — RAG-Powered Codebase Intelligence API
### Full Project Documentation · All 8 Aspects

---

## Overview

**CodeLens** is a production-ready RAG system that lets developers query any GitHub repository in natural language and receive grounded, cited answers with exact file paths and line numbers.

**The problem:** Developers joining a new codebase spend weeks understanding it. Senior engineers waste hours answering the same "how does X work?" questions. No existing tool offers repo-wide natural language comprehension with precise source citations.

**Competitors and the gap:**

| Tool | Limitation |
|---|---|
| GitHub Copilot Chat | Only sees the currently open file |
| Cursor | Great for writing, weak on repo-wide comprehension |
| Sourcegraph Cody | Closest competitor — enterprise-only pricing kills SMB adoption |

**Your wedge:** Open API, per-repo on-demand indexing, B2B SaaS pricing, distributable via GitHub Marketplace or VS Code extension.

**Query types the system handles:**

| Type | Example | Complexity |
|---|---|---|
| Localization | "Where is rate limiting implemented?" | Low |
| Explanation | "What does `TokenRefreshMiddleware` do?" | Medium |
| Bug detection | "Are there unhandled exceptions in the payment flow?" | High |
| Dependency tracing | "What breaks if I change the `User` schema?" | High |

---

## Architecture Diagram

```
GitHub Repo URL
      │
      ▼
┌─────────────────┐
│  Aspect 1       │  Clone repo → filter files → build manifest → upload to S3
│  Data Sourcing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 2       │  Parse (AST) → chunk (function-level) → embed (nomic-embed-code)
│  Data Pipeline  │  → store in Chroma / pgvector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 3       │  Hybrid retrieval (semantic + BM25) → rerank → LLM reasoning
│  Retrieval +    │  → cited answer with file:line references
│  LLM Layer      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 4       │  RAGAS eval: context recall, faithfulness, answer relevance
│  Eval Harness   │  → gated CI deployment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 5       │  FastAPI + Chroma + Prometheus in docker-compose
│  Docker         │  → push image to AWS ECR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 6       │  GitHub Actions: lint → test → build → push → deploy
│  CI/CD          │  GitHub webhook triggers re-index on new commits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 7       │  ECS Fargate + ALB + RDS pgvector + S3
│  AWS Deployment │  Auto-scaling, IAM roles, VPC isolation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aspect 8       │  CloudWatch + Grafana: latency, retrieval hit rate,
│  Monitoring     │  faithfulness drift, user feedback signals
└─────────────────┘
```

---

## Aspect 1 — Problem Definition & Data Sourcing

### What it does
Clones a GitHub repository, filters indexable source files, builds a structured manifest, and mirrors raw files to S3.

### Target repos for development

| Repo | Why |
|---|---|
| `fastapi/fastapi` | Python, clean structure, ~2,400 chunks — good dev size |
| `huggingface/transformers` | Stress-tests scale and retrieval quality |
| `tiangolo/sqlmodel` | Small, good for early debugging |

Use `fastapi/fastapi` as your primary dev repo.

### Code

**Clone / pull:**
```python
import subprocess
from pathlib import Path

def clone_repo(github_url: str, dest: str = "/tmp/repos") -> Path:
    repo_name = github_url.rstrip("/").split("/")[-1]
    dest_path = Path(dest) / repo_name
    if dest_path.exists():
        subprocess.run(["git", "-C", str(dest_path), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", github_url, str(dest_path)], check=True)
    return dest_path
```

**GitHub metadata:**
```python
import requests

def get_repo_metadata(owner: str, repo: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data = requests.get(url, headers=headers).json()
    return {
        "full_name": data["full_name"],
        "description": data["description"],
        "language": data["language"],
        "stars": data["stargazers_count"],
        "default_branch": data["default_branch"],
        "topics": data.get("topics", [])
    }
```

**File filtering:**
```python
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go",
    ".cpp", ".c", ".h", ".rs", ".rb", ".md", ".txt"
}
EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__",
    "dist", "build", ".venv", "venv", "migrations", "vendor"
}
EXCLUDE_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"}

def collect_files(repo_path: Path) -> list[dict]:
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix not in INCLUDE_EXTENSIONS:
            continue
        if path.stat().st_size > 500_000:
            continue
        files.append({
            "abs_path": str(path),
            "rel_path": str(path.relative_to(repo_path)),
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
            "filename": path.name
        })
    return files
```

### S3 folder structure
```
codelens-bucket/
├── repos/
│   └── fastapi__fastapi/
│       ├── raw/                  # original source files, never modified
│       ├── processed/            # cleaned text (Aspect 2 output)
│       ├── chunks/               # chunked + metadata JSONL (Aspect 2 output)
│       └── manifest.json
├── vector_stores/
│   └── fastapi__fastapi/         # per-repo vector DB snapshot
└── indexes/
    └── fastapi__fastapi_v1/
```

**Key design:** one vector store per repo — clean isolation, easier debugging, natural multi-tenant model.

### Manifest schema
```json
{
  "repo": "fastapi/fastapi",
  "indexed_at": "2026-06-26T10:00:00Z",
  "commit_sha": "a3f9c12",
  "total_files": 147,
  "total_size_bytes": 892341,
  "files": [
    {
      "rel_path": "fastapi/routing.py",
      "extension": ".py",
      "size_bytes": 24300,
      "filename": "routing.py"
    }
  ]
}
```

`commit_sha` is your change detection key — used in Aspect 6 to trigger re-indexing via GitHub webhook.

### Outputs
- Cloned repo in `/tmp/repos/fastapi__fastapi/`
- Filtered file list (indexable source files only)
- `manifest.json` uploaded to S3
- Raw files mirrored to `s3://codelens-bucket/repos/fastapi__fastapi/raw/`

---

## Aspect 2 — Data Pipeline (Parsing + Chunking + Embedding)

### Why this is the hardest aspect
Code is not prose. Fixed-token chunking destroys structure — it cuts functions in half, separates docstrings from their function, splits class methods from their class. All retrieval quality flows from getting this right.

### Sub-step 1: AST Parsing with tree-sitter

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def extract_functions(source_code: str) -> list[dict]:
    tree = parser.parse(bytes(source_code, "utf8"))
    root = tree.root_node
    functions = []

    def walk(node, class_name=None):
        if node.type == "class_definition":
            class_name = node.child_by_field_name("name").text.decode()

        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode() if name_node else "unknown"

            body = node.child_by_field_name("body")
            docstring = ""
            if body and body.child_count > 0:
                first = body.children[0]
                if first.type == "expression_statement":
                    inner = first.children[0]
                    if inner.type == "string":
                        docstring = inner.text.decode().strip('"\' ')

            functions.append({
                "name": func_name,
                "class": class_name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": node.text.decode(),
                "docstring": docstring
            })

        for child in node.children:
            walk(child, class_name)

    walk(root)
    return functions
```

### Sub-step 2: Function-level Chunking

Chunking strategy comparison:

| Strategy | How | Good for | Bad for |
|---|---|---|---|
| Function-level | One chunk per function | Precise retrieval, clean citations | Large functions get truncated |
| Fixed token | Every N tokens with overlap | Simple, language-agnostic | Destroys code structure |
| Sliding window | Function + surrounding context | Inter-function dependencies | More chunks, higher cost |

Use function-level as primary. Falls back to fixed-token for non-Python files.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    return len(enc.encode(text))

MAX_CHUNK_TOKENS = 512

def chunk_file(file_info: dict, source_code: str) -> list[dict]:
    ext = file_info["extension"]
    chunks = []

    if ext == ".py":
        functions = extract_functions(source_code)
        for func in functions:
            code = func["code"]
            if estimate_tokens(code) > MAX_CHUNK_TOKENS:
                sub_chunks = fixed_token_split(code, MAX_CHUNK_TOKENS, overlap=64)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(build_chunk(
                        content=sub, file_info=file_info,
                        func_name=func["name"], class_name=func["class"],
                        start_line=func["start_line"],
                        chunk_type="function_part", part_index=i
                    ))
            else:
                chunks.append(build_chunk(
                    content=code, file_info=file_info,
                    func_name=func["name"], class_name=func["class"],
                    start_line=func["start_line"], chunk_type="function"
                ))
    else:
        raw_chunks = fixed_token_split(source_code, MAX_CHUNK_TOKENS, overlap=64)
        for i, text in enumerate(raw_chunks):
            chunks.append(build_chunk(
                content=text, file_info=file_info,
                chunk_type="text_block", part_index=i
            ))

    return chunks

def build_chunk(content, file_info, chunk_type,
                func_name=None, class_name=None,
                start_line=None, part_index=None) -> dict:
    return {
        "content": content,
        "metadata": {
            "repo": file_info["repo"],
            "rel_path": file_info["rel_path"],
            "filename": file_info["filename"],
            "extension": file_info["extension"],
            "func_name": func_name,
            "class_name": class_name,
            "start_line": start_line,
            "chunk_type": chunk_type,
            "part_index": part_index,
            "char_count": len(content)
        }
    }
```

### Sub-step 3: Code-specific Embedding

Use `nomic-ai/nomic-embed-code` — trained on code, free, outperforms ada-002 on code retrieval tasks.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-code")

def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    contents = [c["content"] for c in chunks]
    prefixed = [f"search_document: {c}" for c in contents]  # task prefix required by nomic
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks
```

### Sub-step 4: Vector Store (Chroma local / pgvector prod)

```python
import chromadb, uuid

def store_chunks(chunks: list[dict], repo_id: str):
    client = chromadb.PersistentClient(path=f"./chroma/{repo_id}")
    collection = client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )
```

### Full pipeline callable

```python
def run_pipeline(repo_path: Path, manifest: dict):
    all_chunks = []
    for file_info in manifest["files"]:
        abs_path = repo_path / file_info["rel_path"]
        source_code = abs_path.read_text(errors="ignore")
        file_info["repo"] = manifest["repo"]
        chunks = chunk_file(file_info, source_code)
        all_chunks.extend(chunks)

    all_chunks = embed_chunks(all_chunks)
    repo_id = manifest["repo"].replace("/", "__")
    store_chunks(all_chunks, repo_id)
    upload_vector_store_to_s3(repo_id)
```

### Sanity check
```python
results = collection.query(
    query_texts=["search_query: how does request routing work"],
    n_results=5
)
for r in results["metadatas"][0]:
    print(r["rel_path"], r["func_name"], r["start_line"])
# Should return: fastapi/routing.py, add_api_route, 203
```

### Outputs
- ~2,000–5,000 chunks for a medium repo (fastapi/fastapi: ~2,400)
- Each chunk: full metadata (file path, function name, start line, class, chunk type)
- Chroma DB persisted locally + snapshotted to S3
- Pipeline is idempotent — safe to rerun

---

## Aspect 3 — Retrieval + LLM Layer

### The core logic: how a question becomes an answer

```
User query
    │
    ├──► BM25 keyword search  ─────┐
    │                              ├──► merge → rerank → top-k chunks
    └──► Semantic vector search ───┘
                                         │
                                         ▼
                                   LLM prompt (chunks + query)
                                         │
                                         ▼
                                   Cited answer (file:line refs)
```

### Why hybrid retrieval

Pure semantic search misses exact identifiers. If a user asks "where is `add_api_route` defined?", semantic search might return loosely related chunks. BM25 keyword search catches the exact token. Combining both gives you precision and recall.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_fn = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-code")

def build_retriever(repo_id: str, documents: list):
    # Semantic retriever
    vectorstore = Chroma(
        collection_name="codebase",
        persist_directory=f"./chroma/{repo_id}",
        embedding_function=embedding_fn
    )
    semantic = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Keyword retriever
    keyword = BM25Retriever.from_documents(documents)
    keyword.k = 10

    # Ensemble: 60% semantic, 40% keyword
    return EnsembleRetriever(
        retrievers=[semantic, keyword],
        weights=[0.6, 0.4]
    )
```

### Reranking

After hybrid retrieval you have 20 candidates. Rerank them to get the top 5 that actually matter. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, free, strong.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
```

### LLM prompt construction

The prompt is where citation grounding happens. Each chunk is labelled with its source so the LLM can reference it.

```python
def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        ref = f"{meta['rel_path']}:{meta.get('start_line', '?')}"
        block = f"[SOURCE {i+1}: {ref}]\n```\n{chunk['content']}\n```"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""You are a codebase expert. Answer the developer's question using ONLY the source code provided.
For every claim, cite the source using [SOURCE N] notation.
If you cannot answer from the provided sources, say so explicitly — do not guess.

SOURCES:
{context}

QUESTION: {query}

ANSWER (with citations):"""
```

### LLM call

```python
import anthropic

client = anthropic.Anthropic()

def answer_query(query: str, repo_id: str, documents: list) -> dict:
    retriever = build_retriever(repo_id, documents)
    raw_chunks = retriever.get_relevant_documents(query)
    reranked = rerank(query, [{"content": d.page_content, "metadata": d.metadata}
                               for d in raw_chunks])

    prompt = build_prompt(query, reranked)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {
                "file": c["metadata"]["rel_path"],
                "line": c["metadata"].get("start_line"),
                "function": c["metadata"].get("func_name")
            }
            for c in reranked
        ],
        "query": query,
        "repo": repo_id
    }
```

### Example output
```json
{
  "answer": "Request routing is handled in `fastapi/routing.py`. The `add_api_route` function [SOURCE 1] registers routes by creating an `APIRoute` object and appending it to `self.routes`. Path parameters are extracted via `compile_path` [SOURCE 2].",
  "sources": [
    {"file": "fastapi/routing.py", "line": 203, "function": "add_api_route"},
    {"file": "fastapi/routing.py", "line": 67, "function": "compile_path"}
  ]
}
```

### Outputs
- Hybrid retrieval pipeline (semantic + BM25 + reranker)
- LLM call with grounded prompt
- Structured JSON response with file:line citations
- Latency target: p95 < 3 seconds end-to-end

---

## Aspect 4 — Eval Harness

### Why this exists
Without measurement you are guessing. The eval harness is what separates an MLE project from a demo. It runs automatically in CI and blocks deployment if quality drops.

### Three metrics (RAGAS)

| Metric | What it measures | Target |
|---|---|---|
| Context Recall | Did retrieval pull the right chunks? | > 0.75 |
| Faithfulness | Is the answer grounded in retrieved chunks, not hallucinated? | > 0.85 |
| Answer Relevance | Does the answer address the question? | > 0.80 |

### Build a golden test set

20–30 hand-crafted QA pairs where you know the correct answer and correct source file. This is your ground truth.

```python
GOLDEN_SET = [
    {
        "query": "Where is request routing implemented?",
        "expected_answer": "fastapi/routing.py in add_api_route",
        "expected_sources": ["fastapi/routing.py"],
        "ground_truth_context": "<actual code from routing.py>"
    },
    {
        "query": "How does dependency injection work?",
        "expected_answer": "fastapi/dependencies/utils.py in solve_dependencies",
        "expected_sources": ["fastapi/dependencies/utils.py"],
        "ground_truth_context": "<actual code>"
    },
    # ... 28 more
]
```

### Run RAGAS eval

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def run_eval(golden_set: list, answer_fn) -> dict:
    rows = []
    for item in golden_set:
        result = answer_fn(item["query"])
        rows.append({
            "question": item["query"],
            "answer": result["answer"],
            "contexts": [s["file"] for s in result["sources"]],
            "ground_truth": item["expected_answer"]
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    return scores
```

### CI gate

```python
def check_eval_gate(scores: dict) -> bool:
    THRESHOLDS = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_recall": 0.75
    }
    passed = all(scores[k] >= v for k, v in THRESHOLDS.items())
    if not passed:
        print("EVAL GATE FAILED:")
        for k, v in THRESHOLDS.items():
            status = "PASS" if scores[k] >= v else "FAIL"
            print(f"  {k}: {scores[k]:.3f} (threshold {v}) [{status}]")
    return passed
```

In CI: if `check_eval_gate()` returns `False`, the deployment job does not run.

### Outputs
- 20–30 item golden test set (versioned in git)
- RAGAS scores logged to W&B on every CI run
- Deployment blocked if any metric falls below threshold
- Historical score chart for regression detection

---

## Aspect 5 — Containerization (Docker)

### FastAPI serving layer

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .retrieval import answer_query, load_documents

app = FastAPI(title="CodeLens API", version="1.0.0")

class QueryRequest(BaseModel):
    repo_url: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str

@app.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    repo_id = request.repo_url.rstrip("/").replace("https://github.com/", "").replace("/", "__")
    try:
        documents = load_documents(repo_id)
        result = answer_query(request.question, repo_id, documents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for tree-sitter
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Download embedding model at build time (not runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-code')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - CHROMA_HOST=chroma
    depends_on:
      - chroma
    volumes:
      - ./data:/app/data

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  chroma_data:
```

### requirements.txt

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
anthropic==0.28.0
sentence-transformers==3.0.0
chromadb==0.5.0
langchain==0.2.0
langchain-community==0.2.0
langchain-huggingface==0.0.3
ragas==0.1.9
tree-sitter==0.22.0
tree-sitter-python==0.22.0
tiktoken==0.7.0
boto3==1.34.0
arxiv==2.1.0
PyMuPDF==1.24.0
pydantic==2.7.0
prometheus-client==0.20.0
```

### Local dev workflow

```bash
# Build and run everything
docker compose up --build -d

# Test the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/fastapi/fastapi", "question": "Where is routing implemented?"}'

# Check health
curl http://localhost:8000/health
```

### Smoke test before pushing anywhere

A one-off curl confirms the container *started*; it doesn't confirm the
deploy actually behaves correctly (source_type/path_type filtering, the
LocalStack endpoint override, etc.). Run the smoke test and treat a failure
as a hard stop before the image goes anywhere near a registry:

```bash
python -m scripts.smoke_test
```

It polls `/health` until the container is up, then checks that a code query
and a doc query both return correctly-typed sources with no LLM cost. Exits
non-zero on any failing check, so it can gate CI the same way the retrieval
eval gate does.

### Push to ECR

```bash
aws ecr create-repository --repository-name codelens-api --region us-east-1

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account_id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t codelens-api .
docker tag codelens-api:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/codelens-api:latest
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/codelens-api:latest
```

### Outputs
- `docker compose up` starts the full stack: API + Chroma + Prometheus + Grafana
- Image pushed to ECR, ready for ECS deployment
- Local dev parity with production

---

## Aspect 6 — CI/CD Pipeline

### GitHub Actions workflow

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: codelens-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: |
          pip install ruff
          ruff check app/

      - name: Unit tests
        run: pytest tests/unit/ -v

      - name: Integration tests
        run: pytest tests/integration/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  eval:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run eval harness
        run: python scripts/run_eval.py
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Check eval gate
        run: python scripts/check_gate.py  # exits 1 if below threshold

  deploy:
    needs: [test, eval]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push Docker image
        run: |
          IMAGE_URI=${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}
          docker build -t $IMAGE_URI .
          docker push $IMAGE_URI
          echo "IMAGE_URI=$IMAGE_URI" >> $GITHUB_ENV

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster codelens-cluster \
            --service codelens-api \
            --force-new-deployment \
            --region ${{ env.AWS_REGION }}
```

### GitHub webhook for re-indexing

When a repo you're tracking gets a new commit, re-index it automatically.

```python
# app/webhooks.py
from fastapi import Request, HTTPException
import hmac, hashlib, boto3

WEBHOOK_SECRET = os.environ["GITHUB_WEBHOOK_SECRET"]

@app.post("/webhook/github")
async def github_webhook(request: Request):
    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    body = await request.body()
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    if payload.get("ref") == "refs/heads/main":
        repo_url = payload["repository"]["clone_url"]
        # Trigger async re-index via SQS
        sqs = boto3.client("sqs")
        sqs.send_message(
            QueueUrl=os.environ["REINDEX_QUEUE_URL"],
            MessageBody=json.dumps({"repo_url": repo_url})
        )

    return {"status": "queued"}
```

### Outputs
- Automated: lint → test → eval gate → build → push → deploy on every merge to main
- Eval gate blocks bad deployments automatically
- GitHub webhook triggers re-index on new commits to tracked repos
- Full audit trail in GitHub Actions logs

---

## Aspect 7 — AWS Deployment

### Infrastructure overview

```
Internet
    │
    ▼
┌──────────────────────────────────┐
│  Application Load Balancer (ALB) │
│  HTTPS :443, HTTP → HTTPS        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐  ┌─────────────────────┐
│  ECS Fargate Cluster             │  │  Amazon S3          │
│  codelens-api service            │◄─┤  Raw files          │
│  Task: 1 vCPU, 2GB RAM          │  │  Vector DB snapshots│
│  Min: 2 tasks, Max: 10 tasks    │  │  Manifests          │
└──────────────┬───────────────────┘  └─────────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Amazon RDS (pgvector)           │
│  PostgreSQL 16 + pgvector ext    │
│  db.t3.medium, Multi-AZ          │
└──────────────────────────────────┘
```

### ECS Task Definition (key fields)

```json
{
  "family": "codelens-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "codelens-api",
      "image": "<account_id>.dkr.ecr.us-east-1.amazonaws.com/codelens-api:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "CHROMA_HOST", "value": "pgvector-endpoint"},
        {"name": "AWS_REGION", "value": "us-east-1"}
      ],
      "secrets": [
        {"name": "ANTHROPIC_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "DB_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/codelens-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### RDS pgvector setup

```sql
-- Run once after RDS instance creation
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- nomic-embed-code output dim
    rel_path TEXT,
    func_name TEXT,
    class_name TEXT,
    start_line INTEGER,
    chunk_type TEXT,
    indexed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON code_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX ON code_chunks (repo_id);
```

### Auto-scaling policy

```bash
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/codelens-cluster/codelens-api \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

### SQS re-indexing worker

```python
# worker/reindex_worker.py
import boto3, json, time

sqs = boto3.client("sqs", region_name="us-east-1")
QUEUE_URL = os.environ["REINDEX_QUEUE_URL"]

def poll():
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20  # long polling
        )
        messages = response.get("Messages", [])
        for msg in messages:
            body = json.loads(msg["Body"])
            repo_url = body["repo_url"]
            print(f"Re-indexing: {repo_url}")
            try:
                run_full_pipeline(repo_url)
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except Exception as e:
                print(f"Re-index failed: {e}")
                # Message returns to queue after visibility timeout

if __name__ == "__main__":
    poll()
```

### IAM policy for ECS task role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::codelens-bucket/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage"],
      "Resource": "arn:aws:sqs:us-east-1:*:codelens-reindex-queue"
    },
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:codelens/*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/ecs/codelens-api:*"
    }
  ]
}
```

### Outputs
- Public HTTPS API endpoint via ALB
- ECS Fargate tasks auto-scale based on CPU utilization
- pgvector on RDS for production vector storage
- SQS worker for async re-indexing
- Secrets in Secrets Manager (never in env vars or code)
- Least-privilege IAM task role

---

## Aspect 8 — Production Monitoring

### What to monitor

| Signal | Why | Tool |
|---|---|---|
| API latency (p50/p95/p99) | User experience | CloudWatch / Prometheus |
| Error rate | Reliability | CloudWatch |
| Retrieval hit rate | Are queries being answered? | Custom metric |
| Faithfulness drift | Is answer quality degrading? | Weekly RAGAS run |
| Token usage per query | Cost control | Anthropic API logs |
| Re-index queue depth | Backlog detection | SQS CloudWatch metric |

### Prometheus instrumentation in FastAPI

```python
from prometheus_client import Counter, Histogram, make_asgi_app
import time

REQUEST_COUNT = Counter(
    "codelens_requests_total",
    "Total API requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "codelens_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)
RETRIEVAL_HIT = Counter(
    "codelens_retrieval_hits_total",
    "Queries that returned at least one source"
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())
```

### CloudWatch alarm: high latency

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "codelens-high-latency" \
  --metric-name "codelens_request_duration_seconds" \
  --namespace "CodeLens" \
  --statistic p95 \
  --threshold 5.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --period 60 \
  --alarm-actions arn:aws:sns:us-east-1:...:codelens-alerts
```

### Weekly faithfulness drift check

```python
# scripts/weekly_eval.py — runs as a scheduled ECS task every Sunday

import wandb
from app.eval import run_eval, check_eval_gate
from app.retrieval import answer_query, load_documents

def weekly_check():
    repo_id = "fastapi__fastapi"
    docs = load_documents(repo_id)
    answer_fn = lambda q: answer_query(q, repo_id, docs)

    scores = run_eval(GOLDEN_SET, answer_fn)

    # Log to W&B for historical tracking
    wandb.init(project="codelens-monitoring")
    wandb.log({
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_recall": scores["context_recall"]
    })

    if not check_eval_gate(scores):
        # Send alert to Slack / PagerDuty
        send_alert(f"Weekly eval gate FAILED: {scores}")
```

### Grafana dashboard panels (configure in UI)
- Panel 1: p95 latency over time (line chart, threshold line at 3s)
- Panel 2: Request rate by endpoint (bar chart)
- Panel 3: Error rate % (stat panel, red if > 1%)
- Panel 4: SQS re-index queue depth (line chart)
- Panel 5: Faithfulness score trend (line chart, weekly data points)

### Rollback runbook

```bash
# 1. Identify the last good deployment
aws ecs describe-services --cluster codelens-cluster --services codelens-api

# 2. Roll back to previous task definition revision
aws ecs update-service \
  --cluster codelens-cluster \
  --service codelens-api \
  --task-definition codelens-api:<previous_revision>

# 3. Verify health
watch -n 5 'aws ecs describe-services \
  --cluster codelens-cluster \
  --services codelens-api \
  --query "services[0].runningCount"'

# 4. If vector DB is corrupted, restore from S3 snapshot
aws s3 sync s3://codelens-bucket/vector_stores/fastapi__fastapi_v1/ ./chroma/
```

### Outputs
- Real-time Grafana dashboard: latency, error rate, retrieval hit rate
- CloudWatch alarms → SNS → Slack/PagerDuty for p95 > 5s or error rate > 1%
- Weekly RAGAS faithfulness check logged to W&B
- Rollback runbook that takes < 5 minutes to execute

---

## Summary: What the Finished Product Looks Like

### API surface
```
POST /query          — ask a question about a repo
POST /index          — trigger indexing of a new repo
GET  /repos          — list indexed repos
GET  /health         — health check
POST /webhook/github — GitHub push event handler
GET  /metrics        — Prometheus metrics
```

### Portfolio talking points
- End-to-end RAG with AST-level chunking (not naive text splitting)
- Hybrid retrieval: semantic + BM25 + cross-encoder reranking
- Eval-gated deployment: CI blocks bad code automatically
- Per-repo vector DB isolation (multi-tenant ready)
- Auto re-indexing on commit via GitHub webhook + SQS
- Full observability: Prometheus + Grafana + CloudWatch + W&B

### Skills demonstrated by layer

| Layer | Skills demonstrated |
|---|---|
| Data pipeline | tree-sitter, AST parsing, embedding models, vector DBs |
| Retrieval | hybrid search, reranking, RAG architecture |
| LLM integration | prompt engineering, citation grounding, Anthropic API |
| Evaluation | RAGAS, golden test sets, metric-gated CI |
| Containerization | Docker, docker-compose, ECR |
| CI/CD | GitHub Actions, automated eval gates, webhook integration |
| Cloud | ECS Fargate, ALB, RDS pgvector, S3, SQS, IAM |
| Monitoring | Prometheus, Grafana, CloudWatch, W&B drift detection |

---

*CodeLens — Full Project Documentation · June 2026*
