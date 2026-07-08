"""Pure path/extension classification — no ML, no network, no secrets.

Split out of app/pipeline.py so these rules can be unit tested (and reused)
without pulling in tree-sitter, sentence-transformers, chromadb, or boto3,
all of which app/pipeline.py loads as import-time side effects.
"""

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
