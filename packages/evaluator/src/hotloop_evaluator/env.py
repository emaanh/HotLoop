"""Measurement conditions. No number leaves the evaluator without these (CLAUDE.md)."""

from __future__ import annotations

import os
import platform
import subprocess

import torch

from hotloop_schemas import EnvFingerprint, GpuState


def _smi(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return [x.strip() for x in out.splitlines()[0].split(",")] if out else None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or None


def _int(x: str | None) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def fingerprint(device: str) -> EnvFingerprint:
    env = EnvFingerprint(
        torch=torch.__version__,
        python=platform.python_version(),
        cpu_model=_cpu_model(),
        cuda=torch.version.cuda,
        image=os.environ.get("HOTLOOP_IMAGE"),
        provider=os.environ.get("HOTLOOP_PROVIDER"),
    )
    try:
        import triton

        env.triton = triton.__version__
    except ImportError:
        pass
    if device.startswith("cuda"):
        row = _smi("name,uuid,vbios_version,driver_version,ecc.mode.current")
        if row:
            env.gpu_name, env.gpu_uuid, env.vbios, env.driver = row[:4]
            env.ecc = {"Enabled": True, "Disabled": False}.get(row[4])
    return env


def gpu_state(device: str) -> GpuState | None:
    if not device.startswith("cuda"):
        return None
    row = _smi(
        "clocks.current.sm,clocks.current.memory,temperature.gpu,clocks_throttle_reasons.active"
    )
    if not row:
        return None
    sm, mem, temp, throttle = row
    # bit 0 = "GPU idle", which is benign; anything else means the clocks were being held back
    benign = ("0x0000000000000000", "0x0000000000000001", "[N/A]", "")
    return GpuState(
        sm_clock_mhz=_int(sm),
        mem_clock_mhz=_int(mem),
        temperature_c=_int(temp),
        throttle_reasons=[] if throttle in benign else [throttle],
        clocks_locked=_env_bool("HOTLOOP_CLOCKS_LOCKED"),  # set by whoever locked them (runner)
    )


def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    return None if v is None else v.lower() in ("1", "true", "yes")
