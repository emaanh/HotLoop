import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hotloop_schemas import (
    PUBLIC_FILES,
    Result,
    TaskSpec,
    TrajectoryEvent,
    dump_json_schemas,
    load_task_spec,
    public_content_hash,
)

TASK_TOML = """
schema_version = 1
id = "ragged-attn-a100-0001"
title = "Ragged attention-like, heavy-tailed lengths"
baselines = ["eager", "compile_default", "compile_max_autotune"]

[provenance]
generator_version = "0.0.1"
family = "ragged_attention_like"
seed = 7
sibling_group = "ragged-attn-g3"
[provenance.family_params]
heads = 8
length_dist = "lognormal"

[hardware]
gpu_sku = "A100-SXM4-40GB"
image = "sha256:deadbeef"

[[workload]]
name = "short_heavy_tail"
weight = 0.7
[[workload]]
name = "long_uniform"
weight = 0.3

[outputs]
n_outputs = 1
[[outputs.tolerances]]
rtol = 1e-3
atol = 1e-5

[budget]
max_turns = 100
max_total_tokens = 1500000
gpu_seconds = 1800
wall_seconds = 5400
"""


def _write_task(d: Path) -> Path:
    for name in PUBLIC_FILES:
        (d / name).write_text(TASK_TOML if name == "task.toml" else f"# {name}\n")
    return d


def test_task_roundtrip_and_hash(tmp_path):
    spec = load_task_spec(_write_task(tmp_path))
    assert spec.provenance.sibling_group == "ragged-attn-g3"
    assert [w.name for w in spec.workload] == ["short_heavy_tail", "long_uniform"]

    h = public_content_hash(tmp_path)
    toml = (tmp_path / "task.toml").read_text()
    (tmp_path / "task.toml").write_text(
        toml.replace("schema_version = 1", f'schema_version = 1\ncontent_hash = "{h}"\n')
    )
    assert load_task_spec(tmp_path).content_hash == h
    assert public_content_hash(tmp_path) == h, "hash must be stable once stored in task.toml"
    (tmp_path / "reference.py").write_text("# tampered\n")
    assert public_content_hash(tmp_path) != h


def test_task_rejects_bad_specs(tmp_path):
    base = load_task_spec(_write_task(tmp_path)).model_dump()
    dup = json.loads(json.dumps(base))
    dup["workload"][1]["name"] = dup["workload"][0]["name"]
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(dup)
    tol = json.loads(json.dumps(base))
    tol["outputs"]["n_outputs"] = 2
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(tol)
    extra = json.loads(json.dumps(base))
    extra["surprise"] = 1
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(extra)


def test_result_and_trajectory_minimal():
    r = Result(task_id="t", evaluator_version="0.0.1", status="incorrect", score=0.0)
    assert Result.model_validate_json(r.model_dump_json()) == r
    with pytest.raises(ValidationError):
        Result(task_id="t", evaluator_version="0", status="ok", score=-1)
    e = TrajectoryEvent(run_id="r", seq=0, ts=0.0, turn=0, type="run_start")
    assert TrajectoryEvent.model_validate_json(e.model_dump_json()) == e


def test_committed_json_schemas_are_current(tmp_path):
    committed = Path(__file__).resolve().parents[3] / "schemas"
    for p in dump_json_schemas(tmp_path):
        assert (committed / p.name).read_text() == p.read_text(), (
            f"{p.name} is stale: run `uv run python -m hotloop_schemas`"
        )
