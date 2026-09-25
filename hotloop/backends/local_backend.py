"""Local backend: a GPU machine you control (Lambda, RunPod, AWS, a workstation).

isolation="docker" (default): agent environments and scoring run in containers
  built from docker/Dockerfile, with --network none. Recommended; this is also
  where GPU clocks can be locked (`sudo nvidia-smi -lgc <mhz>`) for stable timing.
isolation="none": plain processes and temp directories. No network or filesystem
  isolation: for development and trusted agents only.

Data lives under `root` with the same layout as the Modal volumes.
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid

from hotloop.bench.store import Store
from hotloop.bench.workspace import prepare_workspace
from hotloop.interface import ExecResult

IMAGE = "hotloop:latest"


def _parse_result(output: str):
    for line in reversed(output.splitlines()):
        if line.startswith("HOTLOOP_RESULT "):
            return json.loads(line[len("HOTLOOP_RESULT "):])
    raise RuntimeError(f"no result in output:\n{output[-3000:]}")


class ProcessEnvironment:
    def __init__(self, workdir: str, gpu: str, env_vars: dict):
        self.workdir = workdir
        self.gpu = gpu
        self.env_vars = env_vars

    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        try:
            p = subprocess.run(["bash", "-lc", command], cwd=self.workdir, env={**os.environ, **self.env_vars},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return ExecResult(p.stdout, p.returncode)
        except subprocess.TimeoutExpired as e:
            return ExecResult((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -1, True)

    def read_text(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def seconds_left(self) -> float:
        return float("inf")

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


class DockerEnvironment:
    def __init__(self, container: str, gpu: str):
        self.container = container
        self.gpu = gpu
        self.workdir = "/workspace"

    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        try:
            p = subprocess.run(["docker", "exec", "-w", self.workdir, self.container, "bash", "-lc", command],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return ExecResult(p.stdout, p.returncode)
        except subprocess.TimeoutExpired as e:
            return ExecResult(str(e.stdout or ""), -1, True)

    def read_text(self, path: str) -> str:
        return subprocess.run(["docker", "exec", self.container, "cat", path], check=True,
                              stdout=subprocess.PIPE, text=True).stdout

    def write_text(self, path: str, content: str) -> None:
        subprocess.run(["docker", "exec", "-i", self.container, "bash", "-c",
                        f"mkdir -p \"$(dirname '{path}')\" && cat > '{path}'"], input=content, text=True, check=True)

    def seconds_left(self) -> float:
        return float("inf")

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class LocalBackend:
    name = "local"

    def __init__(self, root: str = "~/.hotloop", isolation: str = "docker", image: str = IMAGE, gpus: str = "all"):
        root = os.path.abspath(os.path.expanduser(root))
        self.store = Store(os.path.join(root, "tasks"), os.path.join(root, "hidden"), os.path.join(root, "runs"))
        for d in (self.store.tasks, self.store.hidden, self.store.runs):
            os.makedirs(d, exist_ok=True)
        self.isolation = isolation
        self.image = image
        self.gpus = gpus

    # --- helpers -------------------------------------------------------------------
    def _store_env(self) -> dict:
        return {"HOTLOOP_TASKS_DIR": self.store.tasks, "HOTLOOP_HIDDEN_DIR": self.store.hidden,
                "HOTLOOP_RUNS_DIR": self.store.runs}

    def _docker_run(self, args: list[str], stdin: str | None = None, hidden: bool = True, network: bool = False,
                    env: dict | None = None) -> str:
        mounts = ["-v", f"{self.store.tasks}:/vol/tasks", "-v", f"{self.store.runs}:/vol/runs"]
        if hidden:
            mounts += ["-v", f"{self.store.hidden}:/vol/hidden"]
        cmd = ["docker", "run", "--rm", "-i", "--gpus", self.gpus] + ([] if network else ["--network", "none"])
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        p = subprocess.run(cmd + mounts + [self.image] + args, input=stdin, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout

    def _run_module(self, module_args: list[str], stdin: str | None = None, network: bool = False) -> object:
        if self.isolation == "docker":
            env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if network and "HF_TOKEN" in os.environ else None
            out = self._docker_run(["python", "-m"] + module_args, stdin=stdin, network=network, env=env)
        else:
            import sys
            out = subprocess.run([sys.executable, "-m"] + module_args, input=stdin, text=True,
                                 env={**os.environ, **self._store_env()},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        return _parse_result(out)

    # --- agent episodes ----------------------------------------------------------
    def open_environment(self, task_id: str, gpu: str, minutes: float, options: dict | None = None):
        src = self.store.task_dir(task_id)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"unknown task {task_id}")
        if self.isolation == "docker":
            name = f"hotloop-{uuid.uuid4().hex[:10]}"
            subprocess.run(["docker", "run", "-d", "--name", name, "--gpus", self.gpus, "--network", "none",
                            "-v", f"{src}:/task_src:ro", "-w", "/workspace", self.image, "sleep", "infinity"],
                           check=True, stdout=subprocess.DEVNULL)
            env = DockerEnvironment(name, gpu)
            copy = "mkdir -p /workspace/task && cp -r /task_src/* /workspace/task/"
        else:
            wd = tempfile.mkdtemp(prefix="hotloop_ws_")
            env = ProcessEnvironment(wd, gpu, {"HOTLOOP_WORKDIR": wd, "HOTLOOP_TASK_DIR": os.path.join(wd, "task")})
            copy = f"mkdir -p task && cp -r {src}/* task/"
        try:
            res = env.exec(copy, timeout=120)
            if res.exit_code != 0:
                raise RuntimeError(res.output)
            prepare_workspace(env, gpu, minutes, options)
        except Exception:
            env.close()
            raise
        return env

    def score(self, task_id: str, solution_src: str, gpu: str, hidden: bool = True) -> dict:
        args = ["hotloop.harness.score_cli", task_id] + ([] if hidden else ["--public"])
        return self._run_module(args, stdin=solution_src)

    def results(self, since: str = "") -> list[dict]:
        root = os.path.join(self.store.runs, "episodes")
        out = []
        for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            path = os.path.join(root, run, "result.json")
            if run >= since and os.path.exists(path):
                with open(path) as f:
                    out.append(json.load(f))
        return out

    # --- task set ----------------------------------------------------------------
    def list_tasks(self, gpu: str | None = None, kept_only: bool = True) -> list[str]:
        return self.store.kept_tasks(gpu) if gpu and kept_only else self.store.task_ids()

    def select_models(self, n: int) -> list[dict]:
        return self._run_module(["hotloop.gen.cli", "select", "--n", str(n)], network=True)

    def generate(self, model_ids: list[str], phases: tuple[str, ...] = ("prefill", "decode")) -> dict:
        out = self._run_module(["hotloop.gen.cli", "trace"] + model_ids + ["--phases", ",".join(phases)], network=True)
        self._run_module(["hotloop.gen.cli", "annotate"])
        return out

    def filter(self, gpu: str, task_ids: list[str] | None = None) -> dict:
        return self._run_module(["hotloop.gen.cli", "filter"] + (["--tasks", ",".join(task_ids)] if task_ids else []))
