"""Modal server side: images, volumes and deployed functions.

Deploy once (and after code changes) with `hotloop deploy`. Deployed calls keep
running if the local client disconnects.
"""

import modal

from hotloop import config

CUDA_TAG = "12.8.1-devel-ubuntu22.04"

gpu_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "cuda-nsight-compute-12-8", "cuda-nsight-systems-12-8")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "ninja", "transformers", "huggingface_hub", "accelerate")
    .run_commands(
        "useradd -m -u 1500 solver",
        "ln -sf $(ls -d /opt/nvidia/nsight-compute/*/ | head -1)ncu /usr/local/bin/ncu",
        "ln -sf $(ls -d /opt/nvidia/nsight-systems/*/ | head -1)bin/nsys /usr/local/bin/nsys",
    )
    .add_local_python_source("hotloop")
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai>=1.100")
    .add_local_python_source("hotloop")
)

app = modal.App(config.APP_NAME)

tasks_vol = modal.Volume.from_name(config.TASKS_VOLUME, create_if_missing=True)
hidden_vol = modal.Volume.from_name(config.HIDDEN_VOLUME, create_if_missing=True)
runs_vol = modal.Volume.from_name(config.RUNS_VOLUME, create_if_missing=True)
secrets = [modal.Secret.from_name(n) for n in config.SECRET_NAMES]

ALL_VOLUMES = {config.MOUNT_TASKS: tasks_vol, config.MOUNT_HIDDEN: hidden_vol, config.MOUNT_RUNS: runs_vol}


def _store():
    from hotloop.bench.store import Store
    return Store.default()


# --- task set --------------------------------------------------------------------

@app.function(image=gpu_image, secrets=secrets, timeout=1800)
def select_models(n: int) -> list[dict]:
    from hotloop.gen import cli
    return cli.select(n)


@app.function(image=gpu_image, gpu=config.DEV_GPU, secrets=secrets, timeout=3600, volumes=ALL_VOLUMES)
def trace_model(model_id: str, phases: list[str] = ("prefill", "decode")) -> dict:
    from hotloop.gen import cli
    out = cli.trace(_store(), model_id, tuple(phases))
    tasks_vol.commit()
    hidden_vol.commit()
    return out


@app.function(image=gpu_image, gpu=config.DEV_GPU, timeout=3600, volumes=ALL_VOLUMES)
def filter_tasks(task_ids: list[str] | None = None, part: str | None = None) -> dict:
    from hotloop.gen import cli
    out = cli.filter_tasks(_store(), task_ids, part)
    runs_vol.commit()
    return out


@app.function(image=gpu_image, timeout=1800, volumes=ALL_VOLUMES)
def annotate_tasks() -> int:
    from hotloop.gen import cli
    n = cli.annotate(_store())
    tasks_vol.commit()
    return n


@app.function(image=cpu_image, timeout=300, volumes={config.MOUNT_TASKS: tasks_vol, config.MOUNT_RUNS: runs_vol})
def list_tasks(gpu: str | None = None, kept_only: bool = True) -> list[str]:
    store = _store()
    return store.kept_tasks(gpu) if gpu and kept_only else store.task_ids()


@app.function(image=cpu_image, timeout=600, volumes={config.MOUNT_RUNS: runs_vol})
def collect_results(since: str = "") -> list[dict]:
    """result.json of every episode whose run id sorts at or after `since` (e.g. 20260923-1200)."""
    import json
    import os

    root = os.path.join(config.MOUNT_RUNS, "episodes")
    out = []
    for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, run, "result.json")
        if run >= since and os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
    return out


# --- scoring -----------------------------------------------------------------------

@app.function(image=gpu_image, gpu=config.DEV_GPU, timeout=3600, volumes=ALL_VOLUMES)
def score_solution(task_id: str, solution_src: str, hidden: bool = True) -> dict:
    """Score in a fresh container. Hidden shapes never leave this function."""
    import os

    from hotloop.harness.service import run_eval

    os.chmod(config.MOUNT_HIDDEN, 0o700)  # the solution runner drops to an unprivileged user
    result = run_eval(_store(), task_id, solution_src, hidden=hidden)
    runs_vol.commit()
    return result


# --- episodes (orchestrator runs here so API keys stay in Modal) ----------------------

@app.function(image=cpu_image, secrets=secrets, timeout=120)
def preflight_agent(agent: str, agent_kwargs: dict) -> str:
    from hotloop.bench.registry import make_agent

    a = make_agent(agent, **agent_kwargs)
    try:
        return a.preflight() if hasattr(a, "preflight") else "no preflight"
    except Exception as e:
        return f"FAILED: {e!r}"[:500]


@app.function(image=cpu_image, secrets=secrets, timeout=6 * 3600, volumes={config.MOUNT_RUNS: runs_vol})
def run_episode(agent: str, agent_kwargs: dict, task_id: str, gpu: str, minutes: float) -> dict:
    from hotloop.backends.modal_backend import ModalBackend
    from hotloop.bench.episode import run_episode as _run, save_episode
    from hotloop.bench.registry import make_agent

    ep = _run(ModalBackend(), make_agent(agent, **agent_kwargs), task_id, gpu, minutes)
    save_episode(ep, config.MOUNT_RUNS)
    runs_vol.commit()
    return ep["record"]
