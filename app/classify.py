"""Pure path/extension classification and chunk identity — no ML, no
network, no secrets.

Split out of app/pipeline.py so these rules can be unit tested (and reused)
without pulling in tree-sitter, sentence-transformers, chromadb, or boto3,
all of which app/pipeline.py loads as import-time side effects.
"""

import hashlib

# Extensions treated as source code; everything else (.md, .txt) is documentation
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go",
    ".cpp", ".c", ".h", ".rs", ".rb"
}


def get_source_type(extension: str) -> str:
    return "code" if extension in CODE_EXTENSIONS else "doc"


# Where a chunk lives in the repo: library source vs tests vs example snippets vs docs
PATH_TYPE_DIRS = {"tests": "tests", "docs_src": "examples", "docs": "docs"}


def get_path_type(rel_path: str) -> str:
    top = rel_path.replace("\\", "/").split("/")[0]
    return PATH_TYPE_DIRS.get(top, "library")


def chunk_context_header(func_name: str, class_name: str = "", class_docstring: str = "") -> str:
    """A short identifying line prepended to every function/function_part
    chunk's content before embedding -- fixes two confirmed retrieval
    failures, not a speculative improvement:

    1. Split functions (function_part chunks beyond part_index 0) start
       mid-body with no function name or signature text at all, so BM25
       has zero literal overlap with a query naming that function, and the
       embedding lacks framing. Real example: add_api_route's body chunk
       started with "(\\n  generate_unique_id_function..." -- the string
       "add_api_route" appeared nowhere in that chunk's actual text.
    2. FastAPI concentrates natural-language vocabulary in class docstrings
       and Annotated[..., Doc("...")] constructor parameters, while
       behavioral methods (e.g. __call__) are terse code with almost no
       descriptive words -- so a query like "Where is OAuth2 password
       bearer authentication implemented?" has near-zero vocabulary overlap
       with OAuth2PasswordBearer.__call__'s own chunk, even though that's
       exactly where the behavior lives. Confirmed via direct measurement:
       that chunk ranked ~40th of 50 by semantic similarity and didn't
       appear in BM25's top 50 at all.

    Kept to one short line (first sentence of the class docstring, not the
    whole thing) so it primes retrieval without diluting the actual code
    signal or meaningfully changing token counts near MAX_CHUNK_TOKENS.
    """
    qualified = f"{class_name}.{func_name}" if class_name else func_name
    header = f"# {qualified}"
    if class_docstring:
        summary = class_docstring.strip().split("\n")[0].strip()
        if summary:
            header += f" — {summary}"
    return header + "\n"


def chunk_id(metadata: dict) -> str:
    """Deterministic ID from what identifies a chunk's position in the repo
    (not its content/embedding) — so re-running the pipeline on an unchanged
    or lightly-changed file makes Chroma's upsert() genuinely overwrite the
    same record instead of blind-inserting a new one with a random ID every
    time. Without this, every reindex silently doubled the collection
    (found live: fastapi__fastapi held 40,701 chunks against an expected
    ~20k after two full pipeline runs — add_api_route alone had 6 copies
    instead of 2).
    """
    key = ":".join([
        metadata["repo"], metadata["rel_path"], metadata["chunk_type"],
        str(metadata["start_line"]), str(metadata["part_index"]),
    ])
    return hashlib.sha256(key.encode()).hexdigest()
