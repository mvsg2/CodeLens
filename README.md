# CodeLens

CodeLens lets you ask natural-language questions about any GitHub repository and get answers grounded in the actual source code, with file and line citations. Under the hood it's a RAG (retrieval-augmented generation) system: it clones a repo, parses and embeds the code, retrieves the most relevant snippets for a question, and asks an LLM to answer using only those snippets.

Example: point it at `fastapi/fastapi` and ask "How does dependency injection work?" — it finds `fastapi/param_functions.py` and `fastapi/dependencies/utils.py`, and answers with citations to both.

---

## Application Pipeline

There are four stages, and each one solves a different problem:

1. **Sourcing**: clone the target GitHub repo, filter down to indexable source files, record a manifest (file list, commit SHA), and mirror the manifest to S3 (LocalStack locally).
2. **Pipeline (encoding)**: parse each Python file with tree-sitter, chunk it function-by-function (not naive fixed-size splitting, which would cut a function in half), embed every chunk with a code-aware embedding model, and store the vectors + metadata in a Chroma database. This is the slow, GPU-bound step, and it only needs to re-run when the repo has a new commit or the chunking/embedding logic itself changes.
3. **Retrieval**: given a question, search the Chroma index two ways at once (semantic vector search + BM25 keyword search), merge the results, rerank the top candidates with a cross-encoder, and optionally hand the final 5 snippets to an LLM to synthesize a cited answer. This runs on every single question, in seconds.
4. **Eval**: a hand-built set of question/answer pairs with verified correct source files, scored two ways — a free, LLM-free retrieval check (hit rate, MRR), and a RAGAS-based answer-quality check (faithfulness, answer relevancy, context recall) that calls the real LLM. Run the free one after any change to chunking, retrieval, or filtering logic; run the RAGAS one before a release.

Sourcing and the pipeline only need to run again when the repo changes or you change how chunks are built. Retrieval runs per-question. Eval runs whenever you want to check quality.

---

## Local Development

Everything here runs on your own machine. No AWS account needed, no public URL — you index a repo and query it locally.

### Prerequisites

- Python 3.11
- Docker Desktop (for LocalStack, which stands in for AWS S3 locally)
- An NVIDIA GPU is strongly recommended for the embedding step (the pipeline will fall back to CPU, but embedding ~20,000 chunks will be much slower)
- An OpenAI API key (used for the answer-generation LLM call)
- A GitHub personal access token (used to fetch repo metadata — stars, description, default branch)

### Setup

**1. Clone this repo and create a virtual environment**

```bash
git clone <this-repo-url> CodeLens
cd CodeLens
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

PyTorch is not pinned in `requirements.txt` because the right build depends on your GPU. If you have an NVIDIA GPU, install a CUDA-enabled build separately, matching your driver's supported CUDA version, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If you skip this, `sentence-transformers` will fall back to CPU automatically — it'll work, just slower during encoding.

**3. Create a `.env` file in the project root**

```
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your_key_here

# Optional — only needed if you change the LocalStack defaults below
AWS_ENDPOINT_URL=http://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=codelens-bucket
```

**4. Start LocalStack** (mock S3, so you don't need real AWS credentials for local dev)

```bash
docker compose up -d
```

This starts a `localstack` container exposing S3 on `http://localhost:4566`. Manifests and Chroma snapshots get uploaded here instead of real AWS.

### Running everything at once

The simplest way to index a repo and see the whole system work together is:

```bash
python -m scripts.run_all
```

This runs, in order: clone/pull the repo → check whether re-encoding is needed → (re-)encode if so → score retrieval quality against the golden test set → ask a demo question and print the answer with sources.

The first time you run this against a repo, it will re-encode (there's no index yet). On later runs, if the repo hasn't changed and you haven't changed the chunking/embedding code, it will skip straight past encoding — no wasted GPU time.

Useful flags:

```bash
python -m scripts.run_all --repo https://github.com/tiangolo/sqlmodel
# index a different repo instead of the fastapi/fastapi default

python -m scripts.run_all --force-reencode
# re-embed even if nothing changed (e.g. you want a clean rebuild)

python -m scripts.run_all --skip-eval
# skip the quality check step

python -m scripts.run_all --no-llm
# skip the OpenAI call in the demo question — just print which files/functions were retrieved, for free

python -m scripts.run_all --query "Where is authentication handled?"
# ask a different demo question

python -m scripts.run_all --require-gate
# if retrieval quality falls below the passing threshold, stop before spending
# anything on the LLM demo call, instead of just warning and continuing
```

### Running each stage on its own

If you want to inspect or debug a single stage instead of the full run:

```bash
# Stage 1 — clone the repo and build the manifest
python -m scripts.run_sourcing

# Stage 2 — parse, chunk, embed, and store in Chroma (slow, GPU-bound)
python -m scripts.run_pipeline

# Stage 3 — ask sample questions and print LLM-generated answers with citations
python -m scripts.run_retrieval
python -m scripts.run_retrieval --no-llm     # same, but skip the LLM call and just print sources

# Stage 4a — score retrieval quality against the golden test set (no LLM cost)
python -m scripts.run_eval

# Stage 4b — score answer quality with RAGAS: faithfulness, answer relevancy,
# context recall (real LLM cost — one call per item to generate the answer,
# plus several more for RAGAS's own judge model; use --limit to bound cost)
python -m scripts.run_ragas_eval --limit 5
```

The ground truth these two eval stages score against lives in `app/eval.py`'s `GOLDEN_SET` — hand-written, and verifiable yourself with a plain `grep` against the cloned repo (e.g. `grep -n "def add_api_route" -r data/repos/fastapi__fastapi/fastapi/`). The generated *answers* being scored, on the other hand, come from real LLM calls — never canned or precomputed.

Each of these scripts currently targets `fastapi/fastapi` by default (edit the `REPO_URL` / `REPO_ID` constant at the top of the script to point at something else, or use `run_all.py --repo <url>`, which takes it as an argument).

### Running the API server

Once a repo has been indexed at least once (via `run_all.py` or `run_pipeline.py`), you can serve queries over HTTP directly on your host, no Docker involved:

```bash
python -m uvicorn app.main:app --port 8000
```

(To run it containerized instead — the same server, started by Docker rather than by you — see "Verifying a containerized deploy" below.)

Check it's alive:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "repo_url": "https://github.com/fastapi/fastapi",
        "question": "How does dependency injection work?"
      }'
```

The request body supports a few optional fields beyond `repo_url` and `question`:

| Field | Values | Default | What it does |
|---|---|---|---|
| `source_type` | `"code"` / `"doc"` | `"code"` | Search source code chunks, or documentation/markdown chunks |
| `path_type` | `"auto"` / `"library"` / `"tests"` / `"examples"` / `"docs"` | `"auto"` | Restrict retrieval to a part of the repo. `"auto"` searches only the library's own source for code questions, and all docs for doc questions — this avoids test files or tutorial snippets outranking the real implementation |
| `include_answer` | `true` / `false` | `true` | Set to `false` to skip the OpenAI call and get back just the ranked source files — instant and free |

Each entry in the response's `sources` list also carries a `relevance_score` — the cross-encoder's raw relevance score for that chunk against the question. This is a transparency signal, not a calibrated confidence percentage: it's only meaningful *relative to other sources in the same response*, not comparable across different questions. A fixed "below this number = untrustworthy" cutoff was tested against real queries and rejected — a genuinely unanswerable question scored *higher* than a genuinely correct answer to a different question, so no single threshold reliably separates good from bad matches.

Example: ask a documentation question instead of a code question, without spending an LLM call:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "repo_url": "https://github.com/fastapi/fastapi",
        "question": "How do I install FastAPI?",
        "source_type": "doc",
        "include_answer": false
      }'
```

### Verifying a containerized deploy (smoke test)

Once `Dockerfile` and `docker-compose.yml` are wired up, bring up the containerized stack and check it behaves correctly, not just that it started:

```bash
docker compose up --build -d
python -m scripts.smoke_tests.deployment_smoke_test
```

`deployment_smoke_test.py` (in `scripts/smoke_tests/`, alongside the other smoke-test scripts) treats the running API as a black box — it never imports `app/` code, it only makes real HTTP requests to `http://localhost:8000`, the same way a real client would. It polls `/health` until the container is ready, then checks that a code query and a doc query both return correctly-typed sources with no LLM cost.

**Known issue on Windows + Docker Desktop**: during development, `deployment_smoke_test.py` occasionally hit an intermittent `ReadTimeout` on one of the two `/query` calls, even though the container had already answered correctly and fast — confirmed directly from the container's own logs. Multiple fixes were tried (forcing IPv4 over `localhost`, restarting the WSL2/Docker network bridge, switching the script to a shared `requests.Session()`), and none reliably eliminated it. The pattern strongly points to flakiness in Docker Desktop's Windows↔WSL2 network bridge itself, not the application — every request the server received was processed correctly in a couple of milliseconds, with zero exceptions, every single time. This is expected to not reproduce in CI or real AWS deployment, since neither runs through that Windows/WSL2 translation layer at all.

If `deployment_smoke_test.py` hangs on your machine, verify the app itself is actually fine with a few manual `curl` calls instead. Note PowerShell aliases `curl` to `Invoke-WebRequest` — use `curl.exe` to get the real binary:

```powershell
# Code question — expect fastapi/param_functions.py, fastapi/dependencies/utils.py
curl.exe -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{\"repo_url\": \"https://github.com/fastapi/fastapi\", \"question\": \"How does dependency injection work?\", \"include_answer\": false}'

# Code question — expect fastapi/security/*.py
curl.exe -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{\"repo_url\": \"https://github.com/fastapi/fastapi\", \"question\": \"Where is authentication handled?\", \"include_answer\": false}'

# Doc question — expect docs/en/docs/tutorial/index.md
curl.exe -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{\"repo_url\": \"https://github.com/fastapi/fastapi\", \"question\": \"How do I install FastAPI with its standard optional dependencies?\", \"source_type\": \"doc\", \"include_answer\": false}'
```

If these consistently return correct, fast responses, the app is working — the smoke test's occasional hang is environment noise, not a regression.

### Indexing a different repository

The API and CLI both work off a `repo_url`. To index and query a new repo:

```bash
python -m scripts.run_all --repo https://github.com/<owner>/<repo>
```

Then query it with that same URL in `repo_url` — the server converts it internally into a folder name (`owner__repo`) that maps to its own Chroma collection, so different repos never mix data.

---

## Production-Grade Deployment

Not built yet. This section will cover taking CodeLens from the local-only setup above to a real, publicly reachable deployment: automated CI/CD, real AWS infrastructure (ECR, ECS Fargate, an Application Load Balancer, RDS with pgvector, S3, SQS, IAM), a public API endpoint, and production monitoring. See the TODO list below for the current plan.

---

## Project layout

```
.
├── app/
│   ├── sourcing.py       clone/pull a repo, filter files, build + upload the manifest
│   ├── pipeline.py       AST parsing, chunking, embedding, Chroma storage, change-detection state
│   ├── retrieval.py      hybrid search (semantic + BM25), reranking, prompt building, LLM call
│   ├── eval.py           golden test set + retrieval-only scoring (hit rate, MRR) and RAGAS answer-quality scoring
│   ├── classify.py       pure, dependency-free chunk classification/identity (source_type, path_type, chunk_id)
│   ├── scoring.py        pure, dependency-free eval-scoring helpers (answer-gate thresholds, path normalization)
│   ├── main.py           the FastAPI server — exposes /query and /health
│   └── config.py         environment variables and shared constants (e.g. embedding model name)
├── scripts/
│   ├── run_sourcing.py    CLI entry point for the sourcing stage
│   ├── run_pipeline.py    CLI entry point for the encoding stage
│   ├── run_retrieval.py   CLI entry point for asking sample questions
│   ├── run_eval.py        CLI entry point for the free retrieval-only quality gate
│   ├── run_ragas_eval.py  CLI entry point for the RAGAS answer-quality gate (real LLM cost)
│   ├── run_all.py         orchestrates all four stages, skipping encoding when nothing changed
│   ├── inspect_answers.py manual cross-check: generated answers next to retrieved chunks, no judge involved
│   ├── smoke_tests/
│   │   ├── deployment_smoke_test.py     black-box HTTP smoke test for a running containerized deploy
│   │   ├── judge_calibration.py         validates judge output quality against hand-labeled cases
│   │   └── judge_endpoints_smoke_test.py  cheap connectivity check for every registered judge/embedding endpoint
│   └── verifier_tests/
│       ├── validate_golden_set.py       sanity-checks GOLDEN_SET itself before spending money on run_ragas_eval.py
│       └── grep_verify_golden_set.py    AST-based deep check: every expected_functions claim is a real def, reports live line/args
├── worker/
│   └── reindex_worker.py   (planned) background worker that re-indexes a repo when
│                           notified of a new commit, without blocking the API
├── data/                 local artifacts — cloned repos, manifests, Chroma DBs, index state
└── docker-compose.yml    LocalStack (mock S3) for local development
```

---

## TODO

- [x] LLM-based answer-quality evaluation (faithfulness / answer relevancy / context recall via RAGAS — `app/eval.py`'s `evaluate_answers()`, `scripts/run_ragas_eval.py`)
- [x] Containerize the app (Dockerfile) — reindex worker still needs one
- [x] Local "deployment rehearsal": containerized stack running against LocalStack, `scripts/smoke_tests/deployment_smoke_test.py` checks `/health` and `/query`
- [ ] Automated CI/CD (lint → unit tests → integration tests → eval gate → smoke test → build → push → deploy)
- [ ] Migrate the vector store from local Chroma files to pgvector on RDS (needed before running more than one API instance)
- [ ] Real AWS deployment (ECR, ECS Fargate, ALB, RDS, S3, SQS, IAM) and a public API endpoint
- [ ] GitHub webhook → SQS → `worker/reindex_worker.py` for automatic re-indexing on new commits
- [ ] Production monitoring (Prometheus/Grafana, CloudWatch alarms, weekly RAGAS drift check)
