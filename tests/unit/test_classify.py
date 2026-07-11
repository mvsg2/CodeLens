from app.classify import get_source_type, get_path_type, chunk_id, chunk_context_header


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


def _meta(**overrides):
    base = {
        "repo": "fastapi/fastapi", "rel_path": "fastapi/routing.py",
        "chunk_type": "function", "start_line": 2838, "part_index": 0,
    }
    base.update(overrides)
    return base


def test_chunk_id_deterministic_across_calls():
    assert chunk_id(_meta()) == chunk_id(_meta())


def test_chunk_id_differs_by_position():
    # same file, different function (start_line) -> different id
    assert chunk_id(_meta(start_line=2838)) != chunk_id(_meta(start_line=99))


def test_chunk_id_differs_by_part_index():
    # a long function split into parts must get distinct ids per part
    assert chunk_id(_meta(part_index=0)) != chunk_id(_meta(part_index=1))


def test_chunk_id_differs_by_file():
    assert chunk_id(_meta(rel_path="a.py")) != chunk_id(_meta(rel_path="b.py"))


def test_chunk_context_header_module_level_function():
    assert chunk_context_header("jsonable_encoder", "", "") == "# jsonable_encoder\n"


def test_chunk_context_header_method_qualifies_with_class():
    assert chunk_context_header("add_api_route", "APIRouter", "") == "# APIRouter.add_api_route\n"


def test_chunk_context_header_includes_first_line_of_class_docstring():
    header = chunk_context_header(
        "__call__", "OAuth2PasswordBearer",
        "OAuth2 flow for authentication using a bearer token obtained with a password.\n"
        "An instance of it would be used as a dependency.",
    )
    assert header == (
        "# OAuth2PasswordBearer.__call__ — OAuth2 flow for authentication "
        "using a bearer token obtained with a password.\n"
    )


def test_chunk_context_header_ignores_lines_after_the_first():
    # keeps the header short -- must not leak the rest of a multi-paragraph
    # docstring into every method's chunk
    header = chunk_context_header("__call__", "Foo", "Line one.\nLine two.\nLine three.")
    assert "Line two" not in header
    assert "Line three" not in header


def test_chunk_context_header_no_trailing_dash_when_docstring_blank():
    assert chunk_context_header("solve_dependencies", "", "   ") == "# solve_dependencies\n"
