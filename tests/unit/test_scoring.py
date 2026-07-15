from app.scoring import norm_path, is_translation_miss, function_hit, check_answer_gate, ANSWER_QUALITY_THRESHOLDS


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
    scores = {"faithfulness": 1.0}  # answer_relevancy absent
    assert check_answer_gate(scores) is False


def _retrieved(rel_path, class_name, func_name):
    return {"rel_path": rel_path, "class_name": class_name, "func_name": func_name}


def _expected(rel_path, class_name, func_name):
    return {"rel_path": rel_path, "class_name": class_name, "func_name": func_name}


def test_function_hit_true_for_exact_match():
    retrieved = [_retrieved("fastapi/routing.py", "APIRouter", "add_api_route")]
    expected = [_expected("fastapi/routing.py", "APIRouter", "add_api_route")]
    assert function_hit(retrieved, expected) is True


def test_function_hit_false_when_only_file_matches():
    # right file, wrong function -- the exact real gap this metric exists to catch
    retrieved = [_retrieved("fastapi/routing.py", "APIRouter", "get_route_handler")]
    expected = [_expected("fastapi/routing.py", "APIRouter", "add_api_route")]
    assert function_hit(retrieved, expected) is False


def test_function_hit_disambiguates_same_function_name_different_class():
    # add_api_route exists on both APIRouter (routing.py) and FastAPI
    # (applications.py) -- retrieving FastAPI's version shouldn't count as
    # a hit for a query whose ground truth is specifically APIRouter's
    retrieved = [_retrieved("fastapi/applications.py", "FastAPI", "add_api_route")]
    expected = [_expected("fastapi/routing.py", "APIRouter", "add_api_route")]
    assert function_hit(retrieved, expected) is False


def test_function_hit_disambiguates_same_method_name_different_class_same_file():
    # __call__ appears on multiple classes within the same file
    # (OAuth2PasswordBearer vs OAuth2 in oauth2.py) -- class_name must
    # disambiguate, file+func_name alone isn't enough
    retrieved = [_retrieved("fastapi/security/oauth2.py", "OAuth2", "__call__")]
    expected = [_expected("fastapi/security/oauth2.py", "OAuth2PasswordBearer", "__call__")]
    assert function_hit(retrieved, expected) is False


def test_function_hit_true_for_module_level_function_no_class():
    retrieved = [_retrieved("fastapi/encoders.py", "", "jsonable_encoder")]
    expected = [_expected("fastapi/encoders.py", "", "jsonable_encoder")]
    assert function_hit(retrieved, expected) is True


def test_function_hit_true_if_any_of_several_retrieved_chunks_match():
    retrieved = [
        _retrieved("fastapi/routing.py", "APIRouter", "get_route_handler"),
        _retrieved("fastapi/dependencies/utils.py", "", "solve_dependencies"),
    ]
    expected = [_expected("fastapi/dependencies/utils.py", "", "solve_dependencies")]
    assert function_hit(retrieved, expected) is True


def test_function_hit_true_if_any_of_several_expected_functions_match():
    # e.g. a query whose ground truth spans two acceptable functions
    retrieved = [_retrieved("fastapi/encoders.py", "", "jsonable_encoder")]
    expected = [
        _expected("fastapi/routing.py", "", "serialize_response"),
        _expected("fastapi/encoders.py", "", "jsonable_encoder"),
    ]
    assert function_hit(retrieved, expected) is True
