"""Trace a model with Dynamo and cut its graph into per-module tasks.

For every shape, the model's FX graph is captured, real intermediate values are
recorded (as statistics), and each model-defined module (decoder layer,
attention, MLP, norm, ...) is extracted into a standalone PyTorch reference.
Identical modules (layer 0 vs layer 1) are deduplicated structurally.
"""

import hashlib
import json
import operator
import os
import re
import types

import torch
import torch.fx as fx

from hotloop.harness.inputs import make_inputs

# v1 shape grids. The first N_PUBLIC shapes that a model supports are public (agents
# see and bench them); the rest are hidden. Each public set mixes batch 1 and batch > 1,
# short and long lengths, and a non-power-of-two length, so a kernel that only works
# for one case is caught before submission.
# TODO: replace with shapes sampled from public request traces.
N_PUBLIC = 3
MIN_HIDDEN = 2
# prefill: (batch, prompt length)
PREFILL_SHAPES = [(1, 2048), (4, 1024), (3, 777),
                  (1, 512), (2, 4096), (1, 1000), (8, 256), (2, 1536), (1, 3000)]
# decode: (batch, tokens already in the KV cache); one new token per sequence
DECODE_SHAPES = [(16, 2048), (1, 4096), (13, 1500),
                 (8, 1024), (32, 512), (4, 8192), (2, 3000), (24, 777), (64, 256)]
PHASES = ("prefill", "decode")

# Integer/bool inputs (masks, positions, indices) are always stored exactly: replacing
# them with random values changes the task (e.g. a causal mask becomes a full one).
# Float inputs are stored exactly only when they contain inf/nan (additive masks).
EXACT_MAX_NUMEL_FLOAT = 1 << 26
TRIVIAL_TARGETS = {operator.getitem}


def shape_id(batch: int, seq: int) -> str:
    return f"b{batch}_s{seq}"


def shape_meta(phase: str, batch: int, n: int) -> dict:
    """n = prompt length (prefill) or tokens already cached (decode)."""
    if phase == "prefill":
        return {"shape_id": shape_id(batch, n), "phase": phase, "batch": batch, "seq": n, "dims": {"B": batch, "S": n}}
    t = n + 1  # cache length after this step
    return {"shape_id": f"b{batch}_t{t}", "phase": phase, "batch": batch, "seq": 1, "ctx": t,
            "dims": {"B": batch, "T": t}}


def normalize_path(p: str) -> str:
    """Dynamo module paths -> dotted paths ("L['self'].model.layers[0]" -> "model.layers.0")."""
    prev = None
    while prev != p:
        prev = p
        p = re.sub(r"getattr\((.*), ['\"](\w+)['\"]\)", r"\1.\2", p)
    p = re.sub(r"^L\['self'\]", "", p)
    p = re.sub(r"\._modules\['([^']+)'\]", r".\1", p)
    p = re.sub(r"\['([^']+)'\]", r".\1", p)
    p = re.sub(r"\[(\d+)\]", r".\1", p)
    return p.strip(".")


def path_pattern(p: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".*", p)


def module_paths(node: fx.Node) -> list[tuple[str, str]]:
    stack = node.meta.get("nn_module_stack") or {}
    out = []
    for path, cls in stack.values():
        cls_name = cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"
        out.append((normalize_path(path), cls_name))
    return out


def _stats(t: torch.Tensor) -> dict:
    f = t.detach().float()
    finite = f[torch.isfinite(f)]
    if finite.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": finite.mean().item(),
        "std": finite.std().item() if finite.numel() > 1 else 0.0,
        "min": finite.min().item(),
        "max": finite.max().item(),
    }


class Recorder(fx.Interpreter):
    """Runs the graph on real inputs, keeping stats (and small exact copies) per node."""

    def __init__(self, gm, keep: set[str]):
        super().__init__(gm)
        self.keep = keep
        self.values: dict[str, dict] = {}

    def run_node(self, n):
        out = super().run_node(n)
        if n.name in self.keep and isinstance(out, torch.Tensor):
            rec = {"shape": list(out.shape), "dtype": str(out.dtype).removeprefix("torch.")}
            if not out.dtype.is_floating_point:
                rec["exact"] = out.detach().cpu().clone()
            elif bool((~torch.isfinite(out)).any()):
                rec["needs_exact"] = True
                if out.numel() <= EXACT_MAX_NUMEL_FLOAT:
                    rec["exact"] = out.detach().cpu().clone()
            rec["stats"] = _stats(out)
            self.values[n.name] = rec
        return out


def _param_name(ph_name: str, prefix: str) -> str | None:
    s = re.sub(r"^l_", "", ph_name).rstrip("_")
    m = re.match(r"(.*)_(parameters|buffers)_(\w+)$", s)
    if not m:
        return None
    mods = m.group(1).split("_modules_")
    if mods and mods[0] == "self":
        mods = mods[1:]
    pre = prefix.split(".") if prefix else []
    if mods[: len(pre)] == pre:
        mods = mods[len(pre):]
    return "_".join(mods + [m.group(3)])


def _is_param_placeholder(n: fx.Node) -> bool:
    return n.op == "placeholder" and "_parameters_" in n.name


def extract_subgraph(gm: fx.GraphModule, nodes: list[fx.Node], prefix: str):
    """Copy `nodes` into a new graph. External values become placeholders."""
    node_set = set(nodes)
    ext: list[fx.Node] = []
    for n in nodes:
        for a in n.all_input_nodes:
            if a not in node_set and a not in ext:
                ext.append(a)
    # Activations first, then weights.
    ext.sort(key=lambda a: _is_param_placeholder(a))

    g = fx.Graph()
    env, names = {}, {}
    for a in ext:
        pretty = _param_name(a.name, prefix) if a.op == "placeholder" else None
        pretty = re.sub(r"\W", "_", pretty or re.sub(r"^l_", "", a.name).rstrip("_"))
        # HF cache internals -> readable names.
        pretty = re.sub(r"^(kwargs_)?past_key_values_layers_\d+_", "", pretty)
        pretty = {"keys": "key_cache", "values": "value_cache", "cumulative_length": "cache_len"}.get(pretty, pretty)
        if pretty[0].isdigit():
            pretty = "_" + pretty
        ph = g.placeholder(pretty)
        ph.meta = dict(a.meta)
        env[a] = ph
        names[a.name] = ph.name
    for n in nodes:
        env[n] = g.node_copy(n, lambda x: env[x])
    # Drop in-place increments of integer inputs with no other effect (HF's KV-cache
    # position counter): bookkeeping, not kernel work, and it makes the reference
    # non-idempotent (every call would write one slot further into the cache).
    for n in list(g.nodes):
        a0 = n.args[0] if n.args else None
        ev = getattr(a0, "meta", {}).get("example_value") if isinstance(a0, fx.Node) else None
        if (n.op == "call_method" and n.target in ("add_", "sub_") and not n.users and a0.op == "placeholder"
                and isinstance(ev, torch.Tensor) and not ev.dtype.is_floating_point):
            g.erase_node(n)
    outs = [n for n in nodes if any(u not in node_set for u in n.users)]
    g.output(env[outs[0]] if len(outs) == 1 else tuple(env[o] for o in outs))
    g.lint()
    return fx.GraphModule(torch.nn.Module(), g), ext, outs, names


def _import_line(name: str, obj) -> str:
    if isinstance(obj, types.ModuleType):
        return "import torch" if name == "torch" else f"import {obj.__name__} as {name}"
    if name in ("inf", "nan"):
        return f"from math import {name}"
    if obj is torch.device:
        return "from torch import device"
    if obj is type(None):
        return f"{name} = type(None)"
    mod, qual = getattr(obj, "__module__", None), getattr(obj, "__qualname__", "")
    if mod and qual and "." not in qual and "<" not in qual:
        return f"from {mod} import {qual}" + ("" if qual == name else f" as {name}")
    raise ValueError(f"cannot import global {name}={obj!r}")


def to_source(sub_gm: fx.GraphModule, header: str) -> str:
    pc = sub_gm.graph.python_code(root_module="self")
    imports = sorted({_import_line(k, v) for k, v in pc.globals.items() if k != "__builtins__"} | {"import torch"})
    body = pc.src.strip()
    body = re.sub(r"^def forward\(self,?\s*", "def reference(", body)
    return f'"""{header}"""\n\n' + "\n".join(imports) + "\n\n\n" + body + "\n"


def load_reference(src: str):
    ns: dict = {}
    exec(compile(src, "<reference>", "exec"), ns)
    return ns["reference"]


def struct_key(sub_gm: fx.GraphModule, ext: list[fx.Node]) -> str:
    """Hash of the op sequence and weight shapes; ignores activation shapes so the
    same module traced at different sequence lengths gets the same key."""
    ops = [(n.op, str(n.target)) for n in sub_gm.graph.nodes if n.op not in ("placeholder", "output")]
    params = [tuple(a.meta["example_value"].shape) for a in ext if _is_param_placeholder(a)]
    return hashlib.sha1(json.dumps([ops, params], default=str).encode()).hexdigest()[:10]


def _layer_container_paths(model, n_layers: int) -> set[str]:
    """Paths of ModuleLists holding the decoder layers; their ancestors span many
    layers and are skipped (whole-model tasks are out of scope for v1)."""
    out = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers:
            out.add(name)
    return out


def _is_ancestor(p: str, of: str) -> bool:
    return p == "" or of == p or of.startswith(p + ".")


def build_model(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    n = 2 + int(getattr(text_cfg, "first_k_dense_replace", 0) or 0)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        # Enough layers to cover every distinct layer type (e.g. sliding + full attention).
        distinct = list(dict.fromkeys(layer_types))
        n = max(n, max(layer_types.index(t) for t in distinct) + 1)
        n = min(n, 8)
        text_cfg.layer_types = layer_types[:n]
    text_cfg.num_hidden_layers = n
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.cuda().eval(), n


def _capture(model, run) -> list:
    captured = []

    def backend(gm, example_inputs):
        # Snapshot inputs before the graph runs: it may mutate them in place (a KV cache
        # and its position counter), and the Recorder must replay from the pre-step state.
        captured.append((gm, [x.clone() if isinstance(x, torch.Tensor) else x for x in example_inputs]))
        return gm.forward

    torch._dynamo.reset()
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch.no_grad():
        run(compiled)
    return captured


def trace_shape(model, n_layers: int, phase: str, batch: int, n: int, vocab: int, model_id: str) -> dict:
    """Returns {struct_key: task_shape_record} for one phase and shape."""
    meta_base = shape_meta(phase, batch, n)
    if phase == "prefill":
        ids = torch.randint(0, vocab, (batch, n), device="cuda")
        captured = _capture(model, lambda m: m(input_ids=ids, use_cache=False))
    else:
        from transformers import StaticCache

        cache = StaticCache(config=model.config, max_cache_len=n + 1)
        base = getattr(model, model.base_model_prefix)
        with torch.no_grad():  # fill the cache with a real prompt (base model only: no big logits)
            base(input_ids=torch.randint(0, vocab, (batch, n), device="cuda"), past_key_values=cache, use_cache=True)
        ids = torch.randint(0, vocab, (batch, 1), device="cuda")
        captured = _capture(model, lambda m: m(input_ids=ids, past_key_values=cache, use_cache=True))
    return _extract(captured, model, n_layers, model_id, meta_base)


def _extract(captured: list, model, n_layers: int, model_id: str, meta_base: dict) -> dict:
    containers = _layer_container_paths(model, n_layers)
    path_graphs: dict[str, set[int]] = {}
    per_graph = []
    for gi, (gm, _) in enumerate(captured):
        groups: dict[str, list[fx.Node]] = {}
        classes: dict[str, str] = {}
        for node in gm.graph.nodes:
            if node.op not in ("call_function", "call_method", "call_module"):
                continue
            for path, cls in module_paths(node):
                groups.setdefault(path, []).append(node)
                classes[path] = cls
        for p in groups:
            path_graphs.setdefault(p, set()).add(gi)
        per_graph.append((groups, classes))

    results = {}
    for gi, (gm, example_inputs) in enumerate(captured):
        groups, classes = per_graph[gi]
        candidates = []
        for path, nodes in groups.items():
            cls = classes[path]
            if cls.startswith("torch.nn.") or len(path_graphs[path]) > 1:
                continue  # builtin leaf modules (Linear, Embedding) or split by a graph break
            if any(_is_ancestor(path, c) for c in containers):
                continue
            if not any(n.op != "call_function" or n.target not in TRIVIAL_TARGETS for n in nodes):
                continue
            candidates.append(path)
        if not candidates:
            continue

        extracted = {}
        keep = set()
        for path in candidates:
            sub_gm, ext, outs, names = extract_subgraph(gm, groups[path], path)
            extracted[path] = (sub_gm, ext, outs, names)
            keep.update(a.name for a in ext)
        rec = Recorder(gm, keep)
        with torch.no_grad():
            rec.run(*example_inputs)

        for path in candidates:
            sub_gm, ext, outs, names = extracted[path]
            key = struct_key(sub_gm, ext)
            if key in results:
                continue  # identical computation already extracted (layer 1 == layer 0, norm == input_layernorm)
            inputs, exact = [], {}
            ok = True
            for a in ext:
                v = rec.values.get(a.name)
                if v is None:
                    ok = False
                    break
                name = names[a.name]
                if _is_param_placeholder(a):
                    kind = "param"
                elif "exact" in v or v.get("needs_exact") or "_buffers_" in a.name:
                    kind = "exact"
                    exact[name] = v.get("exact")
                    if exact[name] is None:
                        ok = False
                        break
                else:
                    kind = "activation"
                inputs.append({"name": name, "shape": v["shape"], "dtype": v["dtype"], "kind": kind, "stats": v["stats"]})
            if not ok:
                continue
            dims = " ".join(f"{k}={v}" for k, v in meta_base["dims"].items())
            header = (
                f"Auto-generated from {model_id} module `{path}` ({classes[path]}), "
                f"{meta_base['phase']} {dims}.\n\n"
                + "\n".join(f"  {i['name']}: {i['dtype']}{i['shape']} ({i['kind']})" for i in inputs)
            )
            try:
                src = to_source(sub_gm, header)
                meta = {**meta_base, "inputs": inputs}
                meta["outputs"], meta["mutated_inputs"] = validate(src, sub_gm, meta, exact)
            except Exception as e:  # codegen or execution failure: drop, but report
                print(f"[trace] skip {model_id}:{path} ({meta_base['shape_id']}): {type(e).__name__}: {e}")
                continue
            results[key] = {
                "path": path, "module_class": classes[path], "src": src, "meta": meta, "exact": exact,
            }
    return results


def validate(src: str, sub_gm: fx.GraphModule, meta: dict, exact: dict) -> tuple[list[dict], list[int]]:
    """The generated source must reproduce the extracted graph exactly, including any
    in-place updates of its inputs (e.g. writing the new token into a KV cache)."""
    ref = load_reference(src)
    inputs = make_inputs(meta, exact, seed=0)
    ins_a, ins_b = [t.clone() for t in inputs], [t.clone() for t in inputs]
    with torch.no_grad():
        a = ref(*ins_a)
        b = sub_gm(*ins_b)
    mutated = [i for i, (x0, x1) in enumerate(zip(inputs, ins_a)) if not torch.equal(x0, x1)]
    a_flat = torch.utils._pytree.tree_flatten(a)[0] + [ins_a[i] for i in mutated]
    b_flat = torch.utils._pytree.tree_flatten(b)[0] + [ins_b[i] for i in mutated]
    outs = []
    for x, y in zip(a_flat, b_flat, strict=True):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"non-tensor output {type(x)}")
        if not torch.equal(x, y):
            raise ValueError("generated source diverges from graph")
        outs.append({"shape": list(x.shape), "dtype": str(x.dtype).removeprefix("torch.")})
    for i, o in zip(mutated, outs[len(outs) - len(mutated):]):
        o["name"] = meta["inputs"][i]["name"]
        o["in_place"] = True
    return outs, mutated


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model_id.split("/")[-1].lower()).strip("-")


def trace_model(model_id: str, phases=PHASES) -> dict:
    """Returns {task_id: {"task": task_json, "shapes": {sid: record}}} for every phase."""
    model, n_layers = build_model(model_id)
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    vocab = text_cfg.vocab_size
    max_pos = getattr(text_cfg, "max_position_embeddings", None) or 1 << 30
    tasks = {}
    for phase in phases:
        grid = PREFILL_SHAPES if phase == "prefill" else DECODE_SHAPES
        grid = [(b, n) for b, n in grid if n + (phase == "decode") <= max_pos]  # e.g. GPT-2 has 1024 positions
        by_key: dict = {}
        for batch, n in grid:
            sid = shape_meta(phase, batch, n)["shape_id"]
            try:
                res = trace_shape(model, n_layers, phase, batch, n, vocab, model_id)
            except torch.OutOfMemoryError:
                print(f"[trace] OOM {model_id} {phase} {sid}")
                res = {}
            except Exception as e:  # e.g. a model whose cache doesn't support static decode
                print(f"[trace] {model_id} {phase} {sid} failed: {type(e).__name__}: {str(e)[:300]}")
                res = {}
            for key, r in res.items():
                by_key.setdefault(key, {})[sid] = r
            torch.cuda.empty_cache()
        if not grid:
            continue
        order = [shape_meta(phase, *g)["shape_id"] for g in grid]
        for skey, recs in by_key.items():
            traced = [sid for sid in order if sid in recs]
            public, hidden = traced[:N_PUBLIC], traced[N_PUBLIC:]
            if len(public) < N_PUBLIC or len(hidden) < MIN_HIDDEN:
                continue
            first = recs[public[0]]
            pattern = path_pattern(first["path"])
            name = pattern.replace(".*", "").replace("model.", "", 1)
            task_id = f"{slug(model_id)}__{name}" if phase == "prefill" else f"{slug(model_id)}__decode__{name}"
            if task_id in tasks:
                task_id += f"-{skey[:6]}"
            tasks[task_id] = {
                "task": {
                    "task_id": task_id, "model_id": model_id, "module_path": first["path"],
                    "module_class": first["module_class"], "pattern": pattern, "struct_key": skey,
                    "phase": phase, "public_shapes": public, "hidden_shapes": hidden,
                },
                "shapes": recs,
            }
    return tasks


def _dims_of(meta: dict) -> dict:
    return meta.get("dims") or {"B": meta["batch"], "S": meta["seq"]}


def _symbolic_dim(values: list[int], dims: dict[str, list[int]]) -> str:
    for name, vals in dims.items():
        if all(v == d for v, d in zip(values, vals)):
            return name
    names = list(dims)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if all(v == x * y for v, x, y in zip(values, dims[a], dims[b])):
                return f"{a}*{b}"
    if len(set(values)) == 1:
        return str(values[0])
    return "?"


def symbolic_shapes(metas: list[dict]) -> dict:
    """Which input/output dims follow the varying dims (B = batch, S = prompt length,
    T = KV-cache length), from all traced shapes. Tells agents what varies in hidden
    tests without revealing the values."""
    names = list(_dims_of(metas[0]))
    dims = {n: [_dims_of(m)[n] for m in metas] for n in names}
    out = {"dims": names}
    for key in ("inputs", "outputs"):
        specs = []
        for i, first in enumerate(metas[0][key]):
            shape = [_symbolic_dim([m[key][i]["shape"][d] for m in metas], dims) for d in range(len(first["shape"]))]
            specs.append({"name": first.get("name", f"out{i}"), "dtype": first["dtype"], "shape": shape})
        out[key] = specs
    return out


def write_task(task: dict, public_root: str, hidden_root: str):
    t = task["task"]
    t["symbolic_shapes"] = symbolic_shapes([r["meta"] for r in task["shapes"].values()])
    for root, sids in ((public_root, t["public_shapes"]), (hidden_root, t["hidden_shapes"])):
        d = os.path.join(root, t["task_id"])
        os.makedirs(d, exist_ok=True)
        if root == public_root:
            with open(os.path.join(d, "task.json"), "w") as f:
                json.dump({k: v for k, v in t.items() if k != "hidden_shapes"}, f, indent=2)
        else:
            with open(os.path.join(d, "task.json"), "w") as f:
                json.dump(t, f, indent=2)
        for sid in sids:
            r = task["shapes"][sid]
            sd = os.path.join(d, "shapes", sid)
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, "reference.py"), "w") as f:
                f.write(r["src"])
            with open(os.path.join(sd, "meta.json"), "w") as f:
                json.dump(r["meta"], f, indent=2)
            torch.save(r["exact"], os.path.join(sd, "exact.pt"))
