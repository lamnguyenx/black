DEFAULT_EXCLUDES = r"/(\.direnv|\.eggs|\.git|\.hg|\.ipynb_checkpoints|\.mypy_cache|\.nox|\.pytest_cache|\.ruff_cache|\.tox|\.svn|\.venv|\.vscode|__pypackages__|_build|buck-out|build|dist|venv|__submodules__)/"  # noqa: B950
DEFAULT_INCLUDES = r"(\.pyi?|\.ipynb)$"
STDIN_PLACEHOLDER = "__BLACK_STDIN_FILENAME__"

class STYLE_LAMNGUYENX:

    max_line_length = 100

    class no_empty_lines:

        default : int = 3
        opens_block : int = 1
        between_class_and_first_one : int = 1
        between_class_methods : int = 3
        between_short_code : int = 1

    colon_distance_threshold:int = 30
    equal_distance_threshold:int = 25
    equal_after_colon_distance_threshold:int = 20

    class code_height:
        tiny : int = 2
        short : int = 2
        tall : int = 15


class STYLE_DEFAULT:

    max_line_length = 88

    class no_empty_lines:

        default : int = 2
        opens_block : int = 1
        between_class_and_first_one : int = 0
        between_class_methods : int = 1
        between_short_code : int = 1

    colon_distance_threshold:int = 30
    equal_distance_threshold:int = 25
    equal_after_colon_distance_threshold:int = 20

    class code_height:
        tiny : int = 2
        short : int = 2
        tall : int = 15

DEFAULT_LINE_LENGTH = STYLE_DEFAULT.max_line_length
