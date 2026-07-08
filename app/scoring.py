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


ANSWER_QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
}


def check_answer_gate(scores: dict) -> bool:
    return all(scores.get(k, 0) >= v for k, v in ANSWER_QUALITY_THRESHOLDS.items())
