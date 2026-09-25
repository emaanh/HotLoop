"""Modal backend (client side). Talks to the deployed app in modal_app.py."""

import time

import modal

from hotloop import config

RETRIES = 4


def _retry(fn):
    """Transient control-plane errors (a dropped connection to the sandbox) must not
    turn into a lost episode: retry with backoff before giving up."""
    def wrapped(*args, **kwargs):
        for attempt in range(RETRIES):
            try:
                return fn(*args, **kwargs)
            except (ConnectionError, OSError, TimeoutError, modal.exception.ConnectionError,
                    modal.exception.InternalFailure) as e:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 * 3 ** attempt)
    return wrapped
from hotloop.bench.workspace import prepare_workspace
from hotloop.interface import ExecResult


class ModalEnvironment:
    """A Modal sandbox: chosen GPU, no network, public task mounted read-only."""

    def __init__(self, sb, gpu: str):
        self.sb = sb
        self.gpu = gpu
        self.workdir = config.WORKDIR

    @_retry
    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        p = self.sb.exec("bash", "-lc", f"cd {self.workdir} && {{ {command}\n}} 2>&1", timeout=timeout)
        out = p.stdout.read()
        code = p.wait()
        code = p.returncode if code is None else code
        return ExecResult(output=out, exit_code=code, timed_out=code in (-1, 124, 137))

    @_retry
    def read_text(self, path: str) -> str:
        return self.sb.filesystem.read_text(path)

    @_retry
    def write_text(self, path: str, content: str) -> None:
        self.sb.filesystem.write_text(content, path)

    def seconds_left(self) -> float:
        return float("inf")

    def close(self) -> None:
        self.sb.terminate()


class ModalBackend:
    name = "modal"

    def __init__(self, app_name: str = config.APP_NAME):
        self.app_name = app_name

    def _fn(self, name: str):
        return modal.Function.from_name(self.app_name, name)

    # --- agent episodes ----------------------------------------------------------
    def open_environment(self, task_id: str, gpu: str, minutes: float, options: dict | None = None) -> ModalEnvironment:
        from hotloop.backends.modal_app import gpu_image, tasks_vol

        sb = modal.Sandbox.create(
            app=modal.App.lookup(self.app_name, create_if_missing=True),
            image=gpu_image, gpu=gpu, block_network=True, workdir=config.WORKDIR,
            timeout=int(minutes * 60) + 1800,  # hard stop well after the episode deadline
            volumes={config.MOUNT_TASKS: tasks_vol.read_only()},
        )
        env = ModalEnvironment(sb, gpu)
        try:
            res = env.exec(f"mkdir -p task && cp -r {config.MOUNT_TASKS}/{task_id}/* task/", timeout=120)
            if res.exit_code != 0:
                raise RuntimeError(f"could not copy task {task_id}: {res.output}")
            prepare_workspace(env, gpu, minutes, options)
        except Exception:
            env.close()
            raise
        return env

    def score(self, task_id: str, solution_src: str, gpu: str, hidden: bool = True) -> dict:
        return self._fn("score_solution").with_options(gpu=gpu).remote(task_id, solution_src, hidden)

    def score_many(self, jobs: list[tuple[str, str]], gpu: str, hidden: bool = True) -> list:
        fn = self._fn("score_solution").with_options(gpu=gpu)
        return list(fn.starmap([(t, s, hidden) for t, s in jobs], return_exceptions=True))

    # --- remote episodes (orchestrator + API keys live in Modal) --------------------
    def preflight(self, agent: str, agent_kwargs: dict) -> str:
        return self._fn("preflight_agent").remote(agent, agent_kwargs)

    def spawn_episode(self, agent: str, agent_kwargs: dict, task_id: str, gpu: str, minutes: float,
                      options: dict | None = None):
        return self._fn("run_episode").spawn(agent, agent_kwargs, task_id, gpu, minutes, options)

    def results(self, since: str = "") -> list[dict]:
        return self._fn("collect_results").remote(since)

    # --- task set ----------------------------------------------------------------
    def list_tasks(self, gpu: str | None = None, kept_only: bool = True) -> list[str]:
        return self._fn("list_tasks").remote(gpu, kept_only)

    def select_models(self, n: int) -> list[dict]:
        return self._fn("select_models").remote(n)

    def generate(self, model_ids: list[str], phases: tuple[str, ...] = ("prefill", "decode")) -> dict:
        out = {}
        fn = self._fn("trace_model")
        for mid, r in zip(model_ids, fn.starmap([(m, list(phases)) for m in model_ids], return_exceptions=True)):
            out[mid] = r if not isinstance(r, Exception) else {"error": repr(r)[:500]}
        self._fn("annotate_tasks").remote()
        return out

    def filter(self, gpu: str, task_ids: list[str] | None = None, chunks: int = 8) -> dict:
        """Filters run in parallel chunks, each writing its own results file."""
        ids = task_ids or self.list_tasks(kept_only=False)
        parts = [ids[i::chunks] for i in range(chunks) if ids[i::chunks]]
        fn = self._fn("filter_tasks").with_options(gpu=gpu)
        out = {}
        for r in fn.starmap([(p, f"part{i}") for i, p in enumerate(parts)]):
            out.update(r)
        return out
