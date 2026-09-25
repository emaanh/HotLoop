"""Load generated tasks from disk."""

import json
import os
import re
from dataclasses import dataclass, field

import torch


@dataclass
class Shape:
    sid: str
    dir: str
    meta: dict
    src: str | None
    _exact: dict | None = field(default=None, repr=False)

    @property
    def exact(self) -> dict:
        if self._exact is None:
            self._exact = torch.load(os.path.join(self.dir, "exact.pt"))
        return self._exact

    def reference(self, device: str | None = None):
        """The reference function. Traced references hardcode the CUDA device for tensors they
        create (e.g. torch.arange(..., device=...)); `device` retargets them (mps, cpu)."""
        src = self.src
        if device and device != "cuda":
            src = re.sub(r"device\(type='cuda'(, index=\d+)?\)", f"device(type='{device}')", src)
        ns: dict = {}
        exec(compile(src, f"<reference {self.sid}>", "exec"), ns)
        return ns["reference"]


def load_shape(d: str) -> Shape:
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    ref_path = os.path.join(d, "reference.py")
    src = open(ref_path).read() if os.path.exists(ref_path) else None
    return Shape(sid=meta["shape_id"], dir=d, meta=meta, src=src)


def load_task(task_dir: str) -> tuple[dict, list[Shape]]:
    with open(os.path.join(task_dir, "task.json")) as f:
        task = json.load(f)
    sdir = os.path.join(task_dir, "shapes")
    shapes = [load_shape(os.path.join(sdir, s)) for s in sorted(os.listdir(sdir))]
    return task, shapes
