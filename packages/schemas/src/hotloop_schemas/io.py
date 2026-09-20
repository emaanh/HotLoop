from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from hotloop_schemas.result import Result
from hotloop_schemas.task import PUBLIC_FILES, TaskSpec
from hotloop_schemas.trajectory import TrajectoryEvent


def load_task_spec(task_dir: str | Path) -> TaskSpec:
    with open(Path(task_dir) / "task.toml", "rb") as f:
        return TaskSpec.model_validate(tomllib.load(f))


def public_content_hash(task_dir: str | Path) -> str:
    """sha256 over the public files. task.toml is hashed in canonical parsed form without its
    own `content_hash` field, so the hash can live in the file it covers and is insensitive to
    TOML formatting."""
    h = hashlib.sha256()
    for name in PUBLIC_FILES:
        data = (Path(task_dir) / name).read_bytes()
        if name == "task.toml":
            parsed = tomllib.loads(data.decode())
            parsed.pop("content_hash", None)
            data = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        h.update(name.encode() + b"\0" + data + b"\0")
    return h.hexdigest()


def dump_json_schemas(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in (
        ("task.schema.json", TaskSpec),
        ("result.schema.json", Result),
        ("trajectory_event.schema.json", TrajectoryEvent),
    ):
        path = out / name
        path.write_text(json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written
