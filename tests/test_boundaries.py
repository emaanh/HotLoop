"""The four components are decoupled by on-disk contracts (CLAUDE.md, DESIGN §5).

Enforced mechanically: a component may import `hotloop_schemas` and nothing else of ours.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "packages"
COMPONENTS = ["taskgen", "evaluator", "runner", "agents"]


def _hotloop_imports(py: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        found |= {n.split(".")[0] for n in names if n.startswith("hotloop_")}
    return found


def test_components_only_share_schemas():
    violations = []
    for comp in COMPONENTS:
        allowed = {f"hotloop_{comp}", "hotloop_schemas"}
        for py in (ROOT / comp).rglob("*.py"):
            bad = _hotloop_imports(py) - allowed
            if bad:
                violations.append(f"{py.relative_to(ROOT)} imports {sorted(bad)}")
    assert not violations, "\n".join(violations)


def test_schemas_import_nothing_of_ours():
    for py in (ROOT / "schemas").rglob("*.py"):
        assert _hotloop_imports(py) <= {"hotloop_schemas"}, py
