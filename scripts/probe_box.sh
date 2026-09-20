#!/usr/bin/env bash
# Day-1 capability probe of a GPU box (ACCESS.md). Run as the login user with sudo available.
set -u
say() { printf '\n=== %s ===\n' "$*"; }
say "identity"; uname -r; lsb_release -ds 2>/dev/null; nproc; free -g | sed -n 2p
lscpu | grep -E "Model name|Socket|NUMA node\(s\)|Hypervisor"
say "gpu"; nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,memory.total,pstate,persistence_mode,ecc.mode.current,clocks.sm,clocks.max.sm,clocks.max.mem --format=csv
nvidia-smi topo -m 2>/dev/null | head -8
say "profiling restriction"; grep -i RestrictProfiling /proc/driver/nvidia/params || echo "param not found"
say "tools"; for t in nvcc nsys ncu nv-nsight-cu-cli docker python3 pip3 gcc; do printf '%-18s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
nvcc --version 2>/dev/null | tail -2; python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
docker info 2>/dev/null | grep -iE "runtimes|default runtime" ; docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -2
say "persistence mode"; sudo -n nvidia-smi -pm 1; echo "exit=$?"
say "lock gpu clocks"; MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1); sudo -n nvidia-smi -lgc "$MAX,$MAX"; echo "exit=$? (requested $MAX)"
say "lock mem clocks"; MMAX=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1); sudo -n nvidia-smi -lmc "$MMAX,$MMAX"; echo "exit=$? (requested $MMAX)"
say "reset clocks"; sudo -n nvidia-smi -rgc; sudo -n nvidia-smi -rmc; echo done
