"""Shared names and defaults."""

import os

APP_NAME = "hotloop"

# Modal secret holding API keys (HF_TOKEN, OPENAI_API_KEY, ...).
SECRET_NAMES = ["hotloop-keys", "hotloop-vllm"]

# Modal volumes.
TASKS_VOLUME = "hotloop-tasks"    # public task files (visible to agents)
HIDDEN_VOLUME = "hotloop-hidden"  # hidden shapes (scoring only, never visible to agents)
RUNS_VOLUME = "hotloop-runs"      # results, transcripts, caches

# Mount points inside Modal containers (and the local backend's Docker containers).
MOUNT_TASKS, MOUNT_HIDDEN, MOUNT_RUNS = "/vol/tasks", "/vol/hidden", "/vol/runs"

# Where task data lives for code running *inside* a backend machine. Defaults are
# the mount points; the local backend without Docker overrides them.
TASKS_DIR = os.environ.get("HOTLOOP_TASKS_DIR", MOUNT_TASKS)
HIDDEN_DIR = os.environ.get("HOTLOOP_HIDDEN_DIR", MOUNT_HIDDEN)
RUNS_DIR = os.environ.get("HOTLOOP_RUNS_DIR", MOUNT_RUNS)

WORKDIR = "/workspace"  # agent workspace inside containers

# Cheap GPU for development; override per call for real runs.
DEV_GPU = "L4"
