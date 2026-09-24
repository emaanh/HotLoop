"""What a solution may not do.

Three layers:
  1. static check of the source (imports and escape hatches)
  2. TorchDispatchMode: only allocation / view ATen ops may run inside the solution
  3. kernel-name denylist on the profiled CUDA graph (catches library kernels
     launched from C++, cuBLAS, torch.compile output, ...)
"""

import ast
import re

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ALLOWED_IMPORTS = {
    "torch", "triton", "math", "functools", "typing", "dataclasses", "itertools",
    "numpy", "collections", "operator", "enum", "__future__",
}
BANNED_TORCH_MODULES = ("torch._dynamo", "torch._inductor", "torch._export", "torch.fx", "torch._C",
                        "torch.multiprocessing", "torch.distributed", "torch.library", "torch.overrides",
                        "torch.utils._python_dispatch", "torch.profiler")
BANNED_NAMES = {
    "__import__", "eval", "exec", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "open", "breakpoint", "__builtins__",
}
BANNED_ATTRS = {
    "__dict__", "__globals__", "__code__", "__closure__", "__subclasses__", "__bases__",
    "__mro__", "__getattribute__", "f_globals", "f_locals", "f_back", "compile", "_dynamo",
    "modules",  # sys.modules
    # modules re-exported as attributes of torch/triton (torch.os, torch.ctypes, ...)
    "os", "sys", "ctypes", "subprocess", "multiprocessing", "importlib", "builtins", "threading",
    "platform", "_C", "_dynamo", "_inductor", "library",
}
BANNED_CPP = [r"#\s*include\s*<dlfcn\.h>", r"\bdlopen\b", r"\bdlsym\b", r"cublas", r"cudnn", r"\bsystem\s*\(", r"\bpopen\b", r"\bfork\s*\("]


def static_check(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS or m.startswith(BANNED_TORCH_MODULES):
                    problems.append(f"line {node.lineno}: import of '{m}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(f"line {node.lineno}: attribute '{node.attr}' is not allowed")
    for pat in BANNED_CPP:
        m = re.search(pat, src)
        if m:
            problems.append(f"source contains banned pattern '{m.group(0)}'")
    return problems


# ATen ops that only allocate or reinterpret memory. Anything else (math,
# copies, dtype casts) must be the solution's own kernel.
ALLOWED_ATEN = {
    "empty", "empty_strided", "empty_like", "new_empty", "new_empty_strided",
    "zeros", "zeros_like", "new_zeros", "zero_", "fill_",
    "view", "_unsafe_view", "as_strided", "expand", "permute", "transpose", "t",
    "unsqueeze", "squeeze", "slice", "select", "split", "split_with_sizes", "unbind",
    "narrow", "detach", "alias", "lift_fresh", "sym_size", "sym_stride", "sym_numel",
    "sym_storage_offset", "is_contiguous", "view_as_real", "view_as_complex",
    "_reshape_alias", "unflatten", "flatten", "movedim", "diagonal",
    "record_stream", "set_", "resize_", "is_same_size", "_has_compatible_shallow_copy_type",
}


class BanViolation(RuntimeError):
    pass


# Why a banned op usually shows up, and what to do instead. Shown in the violation
# message so every agent gets the same, actionable explanation.
_COPY_HINT = ("a data copy: .reshape()/.contiguous()/.flatten() copy when the tensor is not contiguous "
              "(e.g. after .transpose()/.permute()). Use .view() only on contiguous tensors, or pass the "
              "strides to your kernel and index with them.")
_CAST_HINT = "a dtype/device conversion (.to(), .float(), .half(), .cpu()); do the conversion inside your kernel."
_MATH_HINT = ("a PyTorch compute kernel (tensor arithmetic like x * y or x + 1, a torch function, or an "
              "in-place op); do this computation inside your own kernel.")
BAN_HINTS = {"clone": _COPY_HINT, "copy_": _COPY_HINT, "_to_copy": _CAST_HINT, "to": _CAST_HINT}


class BanMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.violations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if name not in ALLOWED_ATEN:
            msg = (f"torch op aten.{name} called inside solution (only allocation/view ops are allowed). "
                   f"This is {BAN_HINTS.get(name, _MATH_HINT)}")
            self.violations.append(msg)
            raise BanViolation(msg)
        return func(*args, **(kwargs or {}))


# Kernels that come from libraries rather than the solution.
KERNEL_DENY = [
    r"\bat::native::", r"\bat::cuda::", r"\bat_cuda_detail::", r"\bpytorch_flash::",
    r"\bfmha_cutlass", r"^(void )?(sm\d+_xmma|ampere_|volta_|turing_|hopper_|nvjet|cutlass_\d+_)",
    r"cutlass::Kernel2?<cutlass_\d+_", r"\bcublas", r"\bcudnn", r"splitKreduce_kernel",
    r"^triton_(poi|red|per|tem)_", r"\bgemv2?T?_kernel", r"\bgemmk1_kernel", r"\bspin_kernel",
]
_KERNEL_DENY_RE = re.compile("|".join(KERNEL_DENY))


# Memset-like ATen kernels are fine: zero-initialising buffers is allowed.
_KERNEL_ALLOW_RE = re.compile(r"FillFunctor")


def denied_kernels(names: list[str]) -> list[str]:
    return sorted({n for n in names if _KERNEL_DENY_RE.search(n) and not _KERNEL_ALLOW_RE.search(n)})
