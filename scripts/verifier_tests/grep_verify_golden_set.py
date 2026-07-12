"""Deep structural verification of GOLDEN_SET's expected_sources and
expected_functions against the real repo -- stricter than
validate_golden_set.py's basic checks (file-exists, literal `def name(`
substring search).

Uses Python's ast module to actually parse each file's syntax tree,
rather than grep/string matching -- a literal substring search for
"def add_api_route(" would be fooled by the name appearing in a comment,
docstring, or string literal, and can't handle this codebase's frequent
multi-line signatures (the Annotated[..., Doc("...")] pattern) at all.
AST parsing gives an exact, unambiguous answer: does a real
FunctionDef/AsyncFunctionDef node with this name exist, at this class
nesting, with a real parameter list.

Line numbers are discovered and reported, not verified against a stored
value -- GOLDEN_SET's expected_functions doesn't track line numbers (a
deliberate choice: pinning line numbers would make every entry break on
any upstream reformatting, and the (rel_path, class_name, func_name)
triple is already enough to disambiguate). This script's job is
transparency/audit -- showing exactly what's really at that name right
now -- not enforcing a stored line number.

No LLM calls, no cost. Usage:
  python -m scripts.smoke_tests.grep_verify_golden_set
"""
import ast
from pathlib import Path

from app.eval import GOLDEN_SET

REPO_PATH = Path("data/repos/fastapi__fastapi")


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def index_functions(source_code: str) -> dict:
    """Maps (class_name_or_'', func_name) -> {line, args} for every real
    function definition in the file, found via actual AST parsing."""
    tree = ast.parse(source_code)
    index = {}

    def walk(node, class_name=""):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # kwonlyargs (everything after a bare `*`) is a separate list
            # from args/posonlyargs -- missing it undercounts params for
            # any function using keyword-only arguments, a common pattern
            # in this codebase (e.g. solve_dependencies takes 10+ params,
            # all keyword-only, which the first version of this script
            # completely missed and reported as "(no parameters)").
            args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            index[(class_name, node.name)] = {"line": node.lineno, "args": args}
        for child in ast.iter_child_nodes(node):
            walk(child, class_name)

    walk(tree)
    return index


def verify_expected_sources() -> bool:
    print("Verifying expected_sources exist on disk")
    passed = failed = 0
    for item in GOLDEN_SET:
        for source in item["expected_sources"]:
            path = REPO_PATH / source
            ok = path.exists()
            check(source, ok, "" if ok else "file does not exist")
            passed += ok
            failed += not ok
    print(f"  Result: {passed} passed, {failed} failed\n")
    return failed == 0


def verify_expected_functions() -> bool:
    print("Verifying expected_functions via real AST parsing (not grep/string matching)")
    items_with_functions = [i for i in GOLDEN_SET if "expected_functions" in i]

    # Parse each unique file once, not once per function claim
    file_indexes = {}
    for item in items_with_functions:
        for fn in item["expected_functions"]:
            path = REPO_PATH / fn["rel_path"]
            if fn["rel_path"] not in file_indexes and path.exists():
                try:
                    file_indexes[fn["rel_path"]] = index_functions(path.read_text(errors="ignore"))
                except SyntaxError as e:
                    print(f"  FAIL: {fn['rel_path']} failed to parse as Python: {e}")
                    file_indexes[fn["rel_path"]] = None

    passed = failed = 0
    for item in items_with_functions:
        for fn in item["expected_functions"]:
            rel_path, class_name, func_name = fn["rel_path"], fn.get("class_name", ""), fn["func_name"]
            label = f"{rel_path}::{class_name or '(module-level)'}.{func_name}"

            index = file_indexes.get(rel_path)
            if index is None:
                check(label, False, "file missing or failed to parse")
                failed += 1
                continue

            entry = index.get((class_name, func_name))
            if entry is None:
                # Distinguish "function exists but in the wrong class" from
                # "doesn't exist at all" -- more actionable than a flat miss.
                same_name_elsewhere = [k for k in index if k[1] == func_name]
                detail = (f"not found as {class_name or '(module-level)'}.{func_name}; "
                          f"found under: {same_name_elsewhere}" if same_name_elsewhere
                          else "no function with this name anywhere in the file")
                check(label, False, detail)
                failed += 1
                continue

            args_str = ", ".join(entry["args"]) or "(no parameters)"
            check(label, True, f"line {entry['line']}, args: {args_str}")
            passed += 1

    print(f"  Result: {passed} passed, {failed} failed\n")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("GOLDEN SET DEEP VERIFICATION (AST-based)")
    print("=" * 60)
    print(f"Repo: {REPO_PATH}\n")

    sources_ok = verify_expected_sources()
    functions_ok = verify_expected_functions()

    print("=" * 60)
    if sources_ok and functions_ok:
        print("All claims verified against real, current file content.")
    else:
        print("Some claims did not verify -- golden set may be stale "
              "relative to the pinned commit, or contains an error.")
    raise SystemExit(0 if (sources_ok and functions_ok) else 1)
