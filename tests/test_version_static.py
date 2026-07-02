"""Guard that ``openbox_langgraph.__version__`` is a STATIC string literal.

A dynamic version (``importlib.metadata.version(...)``) is forbidden: this SDK's
file instrumentation patches ``open``, and resolving the version per call would
read package METADATA inside a governed ``open`` path — a recursion hazard that
already bit a sibling SDK. The value must also stay in lockstep with the
``[project].version`` in ``pyproject.toml``. Checks are AST/parse based, not
source-grep, so docstrings mentioning the words can't trip them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import openbox_langgraph

_INIT = Path(openbox_langgraph.__file__)
_PYPROJECT = _INIT.parent.parent / "pyproject.toml"


def _version_assignment_node() -> ast.Constant:
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                value = node.value
                assert isinstance(value, ast.Constant), "__version__ must be a literal"
                return value
    raise AssertionError("__version__ assignment not found in openbox_langgraph/__init__.py")


def test_version_is_static_string_literal() -> None:
    node = _version_assignment_node()
    assert isinstance(node.value, str)


def test_runtime_version_matches_literal() -> None:
    node = _version_assignment_node()
    assert openbox_langgraph.__version__ == node.value


def test_version_matches_pyproject() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    assert openbox_langgraph.__version__ == data["project"]["version"]
