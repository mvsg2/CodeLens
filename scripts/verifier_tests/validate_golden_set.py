"""Golden set validation suite — sanity-checks GOLDEN_SET itself before
spending real money running scripts.run_ragas_eval against it. Distinct
from app.eval.evaluate_retrieval(), which checks whether *retrieval*
performs well against the golden set; this checks whether the golden set
*itself* is well-formed and still matches the real repo, reusing
evaluate_retrieval() for the one check where they'd otherwise overlap
rather than reimplementing it.

No LLM calls, no cost -- safe to run any time, same as scripts/run_eval.py.

Usage:
  python -m scripts.smoke_tests.validate_golden_set
"""
import json
from collections import defaultdict
from pathlib import Path

from app.eval import GOLDEN_SET, evaluate_retrieval

REPO_PATH = Path("data/repos/fastapi__fastapi")
CHROMA_PATH = Path("data/chroma/fastapi__fastapi")
REPO_ID = "fastapi__fastapi"

REQUIRED_FIELDS = ["query", "expected_answer", "expected_sources", "source_type", "category"]
DOC_EXTENSIONS = {".md", ".txt", ".rst"}

results = defaultdict(list)


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


# CHECK 1: expected_sources actually exist on disk
def check_source_files_exist() -> bool:
    print("\nCheck 1: expected_sources exist on disk")
    passed = failed = 0
    for item in GOLDEN_SET:
        for source in item["expected_sources"]:
            ok = (REPO_PATH / source).exists()
            if ok:
                passed += 1
            else:
                failed += 1
                results["missing_files"].append(source)
            check(source, ok, f'query: "{item["query"][:60]}"')
    print(f"  Result: {passed} passed, {failed} failed")
    return failed == 0


# CHECK 2: no duplicate queries
def check_no_duplicate_queries() -> bool:
    print("\nCheck 2: no duplicate queries")
    queries = [item["query"] for item in GOLDEN_SET]
    seen, duplicates = set(), []
    for q in queries:
        if q in seen:
            duplicates.append(q)
        seen.add(q)
    for q in duplicates:
        check(f"duplicate: {q}", False)
    ok = check(f"all {len(queries)} queries unique", not duplicates)
    return ok


# CHECK 3: schema -- every item has the real required fields
def check_schema() -> bool:
    print("\nCheck 3: schema validation")
    passed = failed = 0
    for i, item in enumerate(GOLDEN_SET):
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        ok = not missing
        if ok:
            passed += 1
        else:
            failed += 1
        check(f"item {i} ({item.get('query', 'N/A')[:50]!r})", ok,
              f"missing: {missing}" if missing else "")
    print(f"  Result: {passed} passed, {failed} failed")
    return failed == 0


# CHECK 4: category distribution matches GOLDEN_SET's own category field
# (not re-derived from keyword-guessing on the query text, which is a much
# weaker signal than the category the item was actually authored under)
def check_category_distribution() -> bool:
    print("\nCheck 4: category distribution")
    by_category = defaultdict(int)
    for item in GOLDEN_SET:
        by_category[item["category"]] += 1
    for cat, n in sorted(by_category.items()):
        print(f"    {cat:<12} {n}")
    print(f"    {'total':<12} {len(GOLDEN_SET)}")
    return check(f"golden set has {len(GOLDEN_SET)} items (>= 20 recommended)",
                 len(GOLDEN_SET) >= 20)


# CHECK 5: source_type is consistent with the actual file extensions in
# expected_sources -- source_type="code" items shouldn't point at .md/.txt
# files and vice versa. Exempts category="boundary" items deliberately --
# their entire design point is to straddle one code file and one docs file
# under a single source_type label (a hard-filter stress test), not a
# mismatch to flag.
def check_source_type_consistency() -> bool:
    print("\nCheck 5: source_type matches expected_sources' extensions")
    print("  (category=boundary items exempt -- they deliberately mix code+docs sources)")
    passed = failed = 0
    for item in GOLDEN_SET:
        if item["category"] == "boundary":
            continue
        for source in item["expected_sources"]:
            is_doc_ext = Path(source).suffix in DOC_EXTENSIONS
            expected_doc = item["source_type"] == "doc"
            ok = is_doc_ext == expected_doc
            if ok:
                passed += 1
            else:
                failed += 1
                results["source_type_mismatch"].append(source)
            check(f'{source} (source_type={item["source_type"]})', ok)
    print(f"  Result: {passed} passed, {failed} failed")
    return failed == 0


# CHECK 6: expected_functions (hand-verified ground truth) still exist at
# a real def/async def in the right file -- catches the golden set going
# stale if the pinned commit or repo state ever changes, without needing
# an LLM; this is the automated version of the grep verification the
# GOLDEN_SET docstring says was done by hand originally
def check_expected_functions_exist() -> bool:
    print("\nCheck 6: expected_functions exist in source")
    items_with_functions = [i for i in GOLDEN_SET if "expected_functions" in i]
    if not items_with_functions:
        print("  No items have expected_functions -- nothing to check")
        return True

    passed = failed = 0
    for item in items_with_functions:
        for fn in item["expected_functions"]:
            path = REPO_PATH / fn["rel_path"]
            if not path.exists():
                failed += 1
                check(f'{fn["rel_path"]}::{fn["func_name"]}', False, "file does not exist")
                continue
            content = path.read_text(errors="ignore")
            needle = f'def {fn["func_name"]}('
            ok = needle in content
            if ok:
                passed += 1
            else:
                failed += 1
                results["missing_functions"].append(fn)
            check(f'{fn["rel_path"]}::{fn.get("class_name") or "(module-level)"}.{fn["func_name"]}',
                  ok, f'"{needle}" not found in file')
    print(f"  Result: {passed} passed, {failed} failed")
    return failed == 0


# CHECK 7: expected_sources are actually present in the live Chroma index
def check_sources_indexed_in_chroma() -> bool:
    print("\nCheck 7: expected_sources are indexed in Chroma")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection("codebase")
        raw = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"  WARN: could not read Chroma collection ({type(e).__name__}: {e})")
        print(f"  (expected if data/chroma/{REPO_ID} hasn't been indexed yet)")
        return False

    if not raw or not raw.get("metadatas"):
        print("  WARN: Chroma collection is empty")
        return False

    indexed_paths = {m["rel_path"].replace("\\", "/") for m in raw["metadatas"]}
    passed = failed = 0
    for item in GOLDEN_SET:
        for source in item["expected_sources"]:
            ok = source.replace("\\", "/") in indexed_paths
            if ok:
                passed += 1
            else:
                failed += 1
                results["not_indexed"].append(source)
            check(source, ok)
    print(f"  Result: {passed} found, {failed} missing")
    return failed == 0


# CHECK 8: real retrieval quality -- delegates to app.eval.evaluate_retrieval
# rather than reimplementing retrieval here (the original version called
# build_retriever with a signature that no longer exists). Gets both
# file-level and function-level hit rate for free from the tested,
# maintained implementation.
def check_retrieval_quality() -> bool:
    print("\nCheck 8: live retrieval quality (delegates to evaluate_retrieval())")
    try:
        scores = evaluate_retrieval(REPO_ID, top_k=5)
    except Exception as e:
        print(f"  WARN: evaluate_retrieval() failed ({type(e).__name__}: {e})")
        return False

    print(f"    hit_rate:          {scores['hit_rate']:.3f}")
    print(f"    mrr:               {scores['mrr']:.3f}")
    if scores["function_hit_rate"] is not None:
        print(f"    function_hit_rate: {scores['function_hit_rate']:.3f}")
    return check("hit_rate >= 0.75", scores["hit_rate"] >= 0.75, f"got {scores['hit_rate']:.3f}")


def print_summary(check_results: dict):
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    total = len(check_results)
    passed = sum(1 for v in check_results.values() if v)
    for name, result in check_results.items():
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} checks passed")
    if passed == total:
        print("\n  Golden set is valid. Safe to run scripts/run_ragas_eval.py.")
    else:
        print(f"\n  {total - passed} check(s) failed -- review before running scripts/run_ragas_eval.py.")


if __name__ == "__main__":
    print("=" * 60)
    print("CODELENS GOLDEN SET VALIDATION")
    print("=" * 60)
    print(f"Golden set size: {len(GOLDEN_SET)} items")
    print(f"Repo path: {REPO_PATH}")

    check_results = {
        "Source files exist":              check_source_files_exist(),
        "No duplicate queries":            check_no_duplicate_queries(),
        "Schema validation":               check_schema(),
        "Category distribution":           check_category_distribution(),
        "source_type consistency":         check_source_type_consistency(),
        "expected_functions exist":        check_expected_functions_exist(),
        "Sources indexed in Chroma":       check_sources_indexed_in_chroma(),
        "Live retrieval quality":          check_retrieval_quality(),
    }

    print_summary(check_results)

    out = Path("data/validation_report.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed report saved to: {out}")
