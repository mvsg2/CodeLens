from app.classify import get_source_type, get_path_type


def test_source_type_code_extensions():
    for ext in (".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".h", ".rs", ".rb"):
        assert get_source_type(ext) == "code"


def test_source_type_doc_extensions():
    for ext in (".md", ".txt", ".rst", ""):
        assert get_source_type(ext) == "doc"


def test_path_type_library_default():
    assert get_path_type("fastapi/routing.py") == "library"
    assert get_path_type("routing.py") == "library"


def test_path_type_tests():
    assert get_path_type("tests/test_routing.py") == "tests"


def test_path_type_examples():
    assert get_path_type("docs_src/tutorial/first_steps.py") == "examples"


def test_path_type_docs():
    assert get_path_type("docs/en/docs/tutorial/index.md") == "docs"


def test_path_type_windows_separators():
    # rel_path can come from Windows-style pathlib output; must classify
    # the same as the forward-slash form
    assert get_path_type("tests\\test_routing.py") == "tests"
    assert get_path_type("docs\\en\\docs\\index.md") == "docs"
