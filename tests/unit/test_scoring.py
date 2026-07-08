from app.scoring import norm_path, is_translation_miss, check_answer_gate, ANSWER_QUALITY_THRESHOLDS


def test_norm_path_converts_backslashes():
    assert norm_path("docs\\en\\docs\\index.md") == "docs/en/docs/index.md"
    assert norm_path("fastapi/routing.py") == "fastapi/routing.py"


def test_translation_miss_true_for_same_page_different_language():
    assert is_translation_miss(
        "docs/pt/docs/tutorial/index.md",
        "docs/en/docs/tutorial/index.md",
    )


def test_translation_miss_false_for_genuinely_different_page():
    assert not is_translation_miss(
        "docs/en/docs/tutorial/index.md",
        "docs/en/docs/deployment/docker.md",
    )


def test_translation_miss_false_for_non_docs_paths():
    assert not is_translation_miss("fastapi/routing.py", "fastapi/encoders.py")


def test_answer_gate_passes_when_all_thresholds_met():
    scores = {k: v for k, v in ANSWER_QUALITY_THRESHOLDS.items()}
    assert check_answer_gate(scores) is True


def test_answer_gate_fails_when_one_metric_below_threshold():
    scores = {k: v for k, v in ANSWER_QUALITY_THRESHOLDS.items()}
    scores["faithfulness"] = ANSWER_QUALITY_THRESHOLDS["faithfulness"] - 0.01
    assert check_answer_gate(scores) is False


def test_answer_gate_fails_when_metric_missing():
    scores = {"faithfulness": 1.0, "answer_relevancy": 1.0}  # context_recall absent
    assert check_answer_gate(scores) is False
