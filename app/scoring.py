"""Pure eval-scoring helpers — no ML, no network, no secrets.

Split out of app/eval.py so these can be unit tested without pulling in
app.retrieval's transitive heavy imports (tree-sitter, sentence-transformers,
chromadb, boto3, the RAGAS/OpenAI judge call chain).
"""


def norm_path(path: str) -> str:
    return path.replace("\\", "/")


def is_translation_miss(retrieved: str, expected: str) -> bool:
    """True if retrieved and expected are the same doc page in different
    docs languages — docs/en/docs/... vs docs/pt/docs/... — right page,
    wrong language, not a real retrieval miss.
    """
    r, e = norm_path(retrieved).split("/"), norm_path(expected).split("/")
    return (
        len(r) == len(e) and len(r) > 2
        and r[0] == e[0] == "docs" and r[1] != e[1] and r[2:] == e[2:]
    )


def _function_key(rel_path: str, class_name: str, func_name: str) -> tuple[str, str, str]:
    # class_name matters: e.g. add_api_route exists on both APIRouter and
    # FastAPI in different files (file alone disambiguates), but __call__
    # appears multiple times *within the same file* across different
    # classes (OAuth2PasswordBearer.__call__ vs OAuth2.__call__ in the same
    # oauth2.py) -- (file, func_name) alone isn't enough there.
    return (norm_path(rel_path), class_name or "", func_name or "")


def function_hit(retrieved: list[dict], expected_functions: list[dict]) -> bool:
    """True if any retrieved chunk's (rel_path, class_name, func_name) matches
    one of the hand-verified expected (rel_path, class_name, func_name)
    triples. Stricter than file-level hit_rate: retrieving the right file
    but the wrong function no longer counts. `retrieved`/`expected_functions`
    are both lists of {"rel_path", "class_name", "func_name"} dicts
    (class_name may be "" for module-level functions).
    """
    expected_keys = {
        _function_key(e["rel_path"], e.get("class_name", ""), e["func_name"])
        for e in expected_functions
    }
    return any(
        _function_key(r.get("rel_path", ""), r.get("class_name", ""), r.get("func_name", ""))
        in expected_keys
        for r in retrieved
    )


ANSWER_QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
}


def check_answer_gate(scores: dict) -> bool:
    return all(scores.get(k, 0) >= v for k, v in ANSWER_QUALITY_THRESHOLDS.items())
