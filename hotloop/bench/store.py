"""Where tasks, hidden shapes and results live (a directory layout, any filesystem)."""

import json
import os
from dataclasses import dataclass

from hotloop import config


def gpu_key(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


@dataclass
class Store:
    tasks: str    # <tasks>/<task_id>/{task.json, shapes/<sid>/...}   public
    hidden: str   # <hidden>/<task_id>/{task.json, shapes/<sid>/...}  scoring only
    runs: str     # caches, filter results, episodes

    @classmethod
    def default(cls) -> "Store":
        return cls(config.TASKS_DIR, config.HIDDEN_DIR, config.RUNS_DIR)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.runs, "cache")

    def task_ids(self) -> list[str]:
        if not os.path.isdir(self.tasks):
            return []
        return sorted(d for d in os.listdir(self.tasks) if os.path.isfile(os.path.join(self.tasks, d, "task.json")))

    def task_dir(self, task_id: str) -> str:
        return os.path.join(self.tasks, task_id)

    def shape_dirs(self, task_id: str, hidden: bool) -> list[str]:
        roots = [self.tasks] + ([self.hidden] if hidden else [])
        out = []
        for root in roots:
            d = os.path.join(root, task_id, "shapes")
            if os.path.isdir(d):
                out += [os.path.join(d, s) for s in sorted(os.listdir(d))]
        return out

    def filter_path(self, gpu_name: str) -> str:
        return os.path.join(self.cache_dir, f"filters_{gpu_key(gpu_name)}.json")

    def filter_results(self, gpu: str) -> dict:
        """Filter results for a GPU; `gpu` may be a short name ("L4") or a device name."""
        if not os.path.isdir(self.cache_dir):
            return {}
        out = {}
        for f in sorted(os.listdir(self.cache_dir)):
            if f.startswith("filters_") and gpu_key(gpu).lower() in f.lower():
                with open(os.path.join(self.cache_dir, f)) as fh:
                    out.update(json.load(fh))
        return out

    def kept_tasks(self, gpu: str) -> list[str]:
        res = self.filter_results(gpu)
        return [t for t in self.task_ids() if res.get(t, {}).get("keep")]
