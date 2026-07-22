"""Validates the RAGAS faithfulness judge model itself, before trusting it
for the real eval gate — not a test of our retrieval or generation, a test
of the judge's own reliability on hand-labeled cases with unambiguous
correct answers.

Why this exists: investigating why the RAGAS answer-quality gate kept
failing (see notes/eval-harness-and-ragas.md and notes/ci-cd-pipeline.md)
found two real, independent bugs:

  1. app/eval.py's evaluate_answers() never passed an explicit `llm=` to
     ragas_evaluate(), so RAGAS silently defaulted to gpt-3.5-turbo as the
     judge -- a materially weaker model than gpt-4o, which generates the
     actual answers being judged. gpt-3.5-turbo was observed failing to
     even parse RAGAS's own structured-output format for statement
     decomposition ("Failed to parse output. Returning None.").
  2. context_chunks passed to RAGAS stripped the [SOURCE N: path:line]
     headers that the generation prompt (app/retrieval.py's build_prompt)
     actually grounds GPT-4o's answers on -- so the judge was marking
     correctly-cited file paths/line numbers as "unsupported" purely
     because it never saw them. Fixed in app/retrieval.py's source_blocks().

Fixing (2) alone measurably helped (faithfulness on a real example went
0.333 -> 0.667 under a gpt-4o judge). But before adopting any specific
judge model for real, this script checks the judge's verdicts against
cases where the correct answer is not in doubt -- if a judge model can't
get these right, it has no business grading anything subtler.

Two independent things are tested, since they're different mechanisms:
  - DECOMPOSITION_CASES: does the judge even produce valid, parseable
    output for the "break this answer into atomic statements" step?
  - VERIFICATION_CASES: given a context and a single atomic statement,
    does the judge's supported/unsupported verdict match the obviously
    correct answer?

Usage:
  python -m scripts.smoke_tests.judge_calibration
  python -m scripts.smoke_tests.judge_calibration --models gpt-4o
  python -m scripts.smoke_tests.judge_calibration --models gpt-3.5-turbo,gpt-4o
"""
import argparse
import asyncio
import copy
import sys

from ragas.metrics import faithfulness
from ragas.metrics._faithfulness import (
    StatementFaithfulnessAnswers,
    _faithfulness_output_parser,
    _statements_output_parser,
)

from app.eval import JUDGE_MODELS, build_judge

# Every registered candidate by default -- routes through app.eval's
# JUDGE_MODELS registry (build_judge), so OpenRouter-hosted candidates
# (qwen3-8b, gemma-4-31b) work here exactly like OpenAI-hosted ones, using
# whichever API key each needs. This also means this script can validate a
# judge using only OPENROUTER_API_KEY, independent of OpenAI quota/billing
# -- unlike a full scripts.run_ragas_eval run, which still needs whichever
# model app/retrieval.py's GENERATOR_MODEL currently points to for answer
# generation, regardless of which model judges it.
DEFAULT_MODELS = list(JUDGE_MODELS)
# "Near perfect" per the goal of this script: every unambiguous case must
# pass. known_limitation cases are reported but excluded from the gate --
# see VERIFICATION_CASES' citation-marker case for why one exists.
REQUIRED_ACCURACY = 1.0

# ── Verification cases: (context, single atomic statement, expected verdict) ──
# Each case's correct verdict should not be debatable by a competent human
# reader. Mirrors the real [SOURCE N: path:line] format app/retrieval.py's
# source_blocks() produces, so this tests the judge on the actual shape of
# context it sees in production, not a simplified stand-in.
VERIFICATION_CASES = [
    {
        "name": "direct_fact_in_context",
        "context": ["[SOURCE 1: fastapi/routing.py:2838 (add_api_route)]\n"
                    "def add_api_route(self, path, endpoint, ...): ..."],
        "statement": "The add_api_route function is defined in fastapi/routing.py.",
        "expected": 1,
        "note": "File path literally present in the SOURCE header.",
    },
    {
        "name": "direct_fact_absent_from_context",
        "context": ["[SOURCE 1: fastapi/routing.py:2838 (add_api_route)]\n"
                    "def add_api_route(self, path, endpoint, ...): ..."],
        "statement": "The add_api_route function is defined in fastapi/security/oauth2.py.",
        "expected": 0,
        "note": "Wrong file path, not supported by this context at all.",
    },
    {
        "name": "unrelated_topic",
        "context": ["[SOURCE 1: fastapi/background.py:12 (BackgroundTasks)]\n"
                    "class BackgroundTasks: "
                    "collects tasks that Starlette runs after the response is sent."],
        "statement": "FastAPI integrates with Kafka for event streaming.",
        "expected": 0,
        "note": "Context is about background tasks, not messaging systems.",
    },
    {
        "name": "line_number_present",
        "context": ["[SOURCE 1: fastapi/applications.py:1165 (add_api_route)]\n"
                    "def add_api_route(self, path, endpoint, ...): ..."],
        "statement": "add_api_route in applications.py starts at line 1165.",
        "expected": 1,
        "note": ("Regression check for the source_blocks() fix -- line "
                 "number is literally in the SOURCE header."),
    },
    {
        "name": "line_number_wrong",
        "context": ["[SOURCE 1: fastapi/applications.py:1165 (add_api_route)]\n"
                    "def add_api_route(self, path, endpoint, ...): ..."],
        "statement": "add_api_route in applications.py starts at line 42.",
        "expected": 0,
        "note": "Line number contradicts the SOURCE header (1165, not 42).",
    },
    {
        "name": "multi_chunk_synthesis",
        "context": [
            "[SOURCE 1: fastapi/routing.py:2838 (add_api_route)]\n"
            "def add_api_route(self, path, endpoint, ...): ...",
            "[SOURCE 2: fastapi/applications.py:1165 (add_api_route)]\n"
            "def add_api_route(self, path, endpoint, ...): ...",
        ],
        "statement": "add_api_route is defined in both fastapi/routing.py and fastapi/applications.py.",
        "expected": 1,
        "note": "Fact spans two separate context chunks -- judge must synthesize, not just pattern-match one.",
    },
    {
        "name": "paraphrase_not_literal_quote",
        "context": ["[SOURCE 1: fastapi/security/oauth2.py:40 (OAuth2PasswordBearer)]\n"
                    "class OAuth2PasswordBearer:\n"
                    "    def __call__(self, request):\n"
                    "        if not authorization or scheme.lower() != 'bearer':\n"
                    "            if self.auto_error:\n"
                    "                raise HTTPException(status_code=401, ...)"],
        "statement": "OAuth2PasswordBearer raises a 401 error when no valid bearer token is present.",
        "expected": 1,
        "note": ("Correct claim phrased in different words than the code -- "
                 "judge shouldn't require a literal quote match."),
    },
    {
        "name": "citation_marker_self_reference",
        "context": ["[SOURCE 1: fastapi/routing.py:2838 (add_api_route)]\n"
                    "def add_api_route(self, path, endpoint, ...): ..."],
        "statement": "This information is from SOURCE 1.",
        "expected": 1,
        "note": ("KNOWN LIMITATION, excluded from the pass/fail gate: the "
                 "judge tends to mark self-referential citation-marker "
                 "statements as unsupported/'too vague' even when the "
                 "citation is correct -- observed directly in real runs. "
                 "Tracked here so a future judge-model change is measured "
                 "against this known gap, not silently re-discovered."),
        "known_limitation": True,
    },
]

# ── Decomposition cases: full multi-sentence answers, checking the judge
# can even produce valid structured output for the first pipeline step ──
DECOMPOSITION_CASES = [
    {
        "name": "two_location_answer",
        "question": "Where is `add_api_route` defined?",
        "answer": (
            "The `add_api_route` function is defined in two places:\n"
            "1. **File:** `fastapi/applications.py`, **Function:** `add_api_route` [SOURCE 1].\n"
            "2. **File:** `fastapi/routing.py`, **Function:** `add_api_route` [SOURCE 2]."
        ),
        "min_statements": 2,
    },
    {
        "name": "explanation_answer",
        "question": "How does dependency injection resolve nested dependencies?",
        "answer": (
            "Dependency injection resolves nested dependencies through the "
            "`solve_dependencies` function in `fastapi/dependencies/utils.py`. "
            "It walks the dependant tree recursively, awaiting each "
            "sub-dependency and caching results to avoid redundant calls "
            "within the same request [SOURCE 1]."
        ),
        "min_statements": 2,
    },
]


async def judge_with(model_name: str, row: dict, statement: str) -> int | None:
    metric = copy.copy(faithfulness)
    metric.llm = build_judge(model_name)
    p_value = metric._create_nli_prompt(row, [statement])
    resp = await metric.llm.generate(p_value, is_async=True, n=1)
    parsed = await _faithfulness_output_parser.aparse(
        resp.generations[0][0].text, p_value, metric.llm, metric.max_retries
    )
    if parsed is None:
        return None
    verdicts = StatementFaithfulnessAnswers.parse_obj(parsed.dicts())
    if not verdicts.__root__:
        return None
    return verdicts.__root__[0].verdict


async def decompose_with(model_name: str, row: dict) -> int | None:
    metric = copy.copy(faithfulness)
    metric.llm = build_judge(model_name)
    p_value = metric._create_statements_prompt(row)
    resp = await metric.llm.generate(p_value, is_async=True)
    parsed = await _statements_output_parser.aparse(
        resp.generations[0][0].text, p_value, metric.llm, metric.max_retries
    )
    if parsed is None:
        return None
    statements = [s for item in parsed.dicts() for s in item["simpler_statements"]]
    return len(statements)


def run_verification_cases(model_name: str) -> tuple[int, int, list[str]]:
    passed, total, failures = 0, 0, []
    for case in VERIFICATION_CASES:
        row = {"question": "", "contexts": case["context"]}
        actual = asyncio.run(judge_with(model_name, row, case["statement"]))
        ok = actual == case["expected"]
        gated = not case.get("known_limitation", False)
        if gated:
            total += 1
            if ok:
                passed += 1
        tag = "[KNOWN LIMITATION]" if case.get("known_limitation") else ""
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {case['name']} {tag} (expected={case['expected']}, got={actual})")
        if not ok:
            failures.append(f"{case['name']}: {case['note']}")
    return passed, total, failures


def run_decomposition_cases(model_name: str) -> tuple[int, int, list[str]]:
    passed, total, failures = 0, len(DECOMPOSITION_CASES), []
    for case in DECOMPOSITION_CASES:
        row = {"question": case["question"], "answer": case["answer"]}
        n = asyncio.run(decompose_with(model_name, row))
        ok = n is not None and n >= case["min_statements"]
        status = "PASS" if ok else "FAIL"
        detail = f"{n} statements" if n is not None else "FAILED TO PARSE"
        print(f"    [{status}] {case['name']} ({detail}, need >= {case['min_statements']})")
        if ok:
            passed += 1
        else:
            failures.append(f"{case['name']}: {detail}")
    return passed, total, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated candidate judge models to test")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    print("Validating the RAGAS judge model itself against hand-labeled cases.")
    print("This does NOT test retrieval or generation quality — only whether the")
    print("judge's own verdicts are reliable enough to trust for the real gate.\n")

    overall_gate = True
    for model in models:
        print("=" * 70)
        print(f"CANDIDATE JUDGE MODEL: {model}")
        print("=" * 70)

        print("\n  -- decomposition (does the judge even parse this step?) --")
        d_passed, d_total, d_failures = run_decomposition_cases(model)

        print("\n  -- verification (is the supported/unsupported verdict correct?) --")
        v_passed, v_total, v_failures = run_verification_cases(model)

        d_acc = d_passed / d_total if d_total else 0.0
        v_acc = v_passed / v_total if v_total else 0.0
        model_gate = d_acc >= REQUIRED_ACCURACY and v_acc >= REQUIRED_ACCURACY

        print(f"\n  Decomposition: {d_passed}/{d_total} ({d_acc:.0%})")
        print(f"  Verification:  {v_passed}/{v_total} ({v_acc:.0%}, known-limitation cases excluded)")
        print(f"  GATE: {'PASS' if model_gate else 'FAIL'} (required {REQUIRED_ACCURACY:.0%})")
        if not model_gate:
            overall_gate = False
            for f in d_failures + v_failures:
                print(f"    - {f}")
        print()

    sys.exit(0 if overall_gate else 1)


if __name__ == "__main__":
    main()
