"""Committed task packages must match their own content hash.

Regression test: a repo-wide `ruff format` once rewrote generated workload.py files after they
were sealed; the evaluator (correctly) refused them and an agent burned ten turns on it.
"""

from pathlib import Path

import pytest

from hotloop_schemas import PRIVATE_DIR, load_task_spec, public_content_hash

PACKAGES = sorted(
    p.parent for p in (Path(__file__).resolve().parents[1] / "benchmark").glob("*/*/task.toml")
)


@pytest.mark.parametrize("pkg", PACKAGES, ids=lambda p: p.name)
def test_package_matches_its_seal(pkg):
    spec = load_task_spec(pkg)
    assert spec.content_hash, "committed packages must be sealed"
    assert spec.content_hash == public_content_hash(pkg)
    assert not (pkg / PRIVATE_DIR).exists(), "private material must never be committed"


def test_there_are_packages():
    assert len(PACKAGES) >= 12
