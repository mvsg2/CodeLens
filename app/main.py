from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.retrieval import answer_query

app = FastAPI(title="CodeLens API", version="1.0.0")


class QueryRequest(BaseModel):
    repo_url: str
    question: str
    source_type: Literal["code", "doc"] = "code"
    # "auto" = library source for code queries, all docs for doc queries
    path_type: Literal["auto", "library", "tests", "examples", "docs"] | None = "auto"
    # False = skip the LLM call and return sources only (faster, free)
    include_answer: bool = True


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[dict]
    query: str
    repo: str


def repo_id_from_url(repo_url: str) -> str:
    return (
        repo_url.rstrip("/")
        .replace("https://github.com/", "")
        .replace("/", "__")
    )


@app.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    repo_id = repo_id_from_url(request.repo_url)
    try:
        return answer_query(request.question, repo_id,
                            source_type=request.source_type,
                            path_type=request.path_type,
                            include_answer=request.include_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
