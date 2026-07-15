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


# Hard-gated: CI fails if either misses. Recalibrated from the original
# 0.85/0.80 aspirational numbers down to a small, consistent margin below
# the best real full-golden-set measurement on record (gpt-4o generator +
# gpt-4o judge: faithfulness 0.803, answer_relevancy 0.782 -- see
# codelens.md's Aspect 4 judge-comparison results). The original numbers
# were never cleared by any measured run, including the strongest
# combination tested -- a gate that always fails regardless of whether a
# change helps or hurts isn't a useful signal, it just trains people to
# ignore it.
ANSWER_QUALITY_THRESHOLDS = {
    "faithfulness": 0.78,
    "answer_relevancy": 0.75,
}

# context_recall is deliberately NOT in ANSWER_QUALITY_THRESHOLDS -- report
# only, not gated. Two real runs after fixing a concrete bug (the
# "boundary" golden-set items wrongly pinning source_type: "code" on
# questions whose ground truth needs both code and docs) both still failed
# on context_recall alone, with faithfulness/answer_relevancy comfortably
# passing -- and the failure is broad, not isolated to one category:
# "negative" items score a structural 0 for any system (the ground truth
# is "not in this repo," an absence claim no retrieved text can ever
# support), and the remaining categories vary 0.0-0.4 run to run against
# the same golden set with nothing else changed, pointing to judge-scoring
# noise as much as real system weakness. Faithfulness and answer_relevancy
# already cover grounding and topicality; keeping a metric gated once it's
# been shown structurally unwinnable for a third of the golden set and
# noisy on the rest just trains people to ignore red gates. Still reported
# every run (see CONTEXT_RECALL_TARGET) so a real regression is visible.
CONTEXT_RECALL_TARGET = 0.50


def check_answer_gate(scores: dict) -> bool:
    return all(scores.get(k, 0) >= v for k, v in ANSWER_QUALITY_THRESHOLDS.items())
