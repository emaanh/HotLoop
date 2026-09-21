import importlib.util
import itertools

import pytest
import torch

from hotloop_schemas import load_task_spec, public_content_hash
from hotloop_taskgen.core import emit
from hotloop_taskgen.families import FAMILIES

ALL = [(f, r) for f, mod in FAMILIES.items() for r in mod.REGIMES]
HW = {"gpu_sku": "cpu-test", "image": "none"}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("family,regime", ALL)
def test_every_regime_emits_a_sealed_valid_package(tmp_path, family, regime):
    d = emit(FAMILIES[family].draft(regime), tmp_path, HW, calibrate=False)
    spec = load_task_spec(d)
    assert spec.content_hash == public_content_hash(d)
    assert spec.provenance.family == family and spec.provenance.family_params["regime"] == regime
    assert spec.provenance.sibling_group
    readme = (d / "AGENT_README.md").read_text()
    assert "solution.py" in readme and "fastest automatic baseline" in readme
    # nothing about strategies or expected answers may leak into what the agent sees
    for leak in ("multi_dot", "jagged", "scatter", "headroom", "control"):
        assert leak not in readme, leak
    with pytest.raises(FileExistsError):
        emit(FAMILIES[family].draft(regime), tmp_path, HW, calibrate=False)


@pytest.mark.parametrize("regime", list(FAMILIES["ragged_pool"].REGIMES))
def test_ragged_lengths_have_fixed_total_for_every_seed(tmp_path, regime):
    d = emit(FAMILIES["ragged_pool"].draft(regime), tmp_path, HW, calibrate=False)
    wl = _load(d / "workload.py", f"wl_{regime}")
    seen = set()
    for seed in (0, 1, 987654321):
        lens = wl.lengths(seed)
        assert int(lens.sum()) == wl.T and lens.numel() == wl.S
        assert 1 <= int(lens.min()) and int(lens.max()) <= wl.LMAX
        seen.add(tuple(lens[:64].tolist()))
    if wl.DIST["kind"] != "uniform":
        assert len(seen) > 1, "length pattern must change with the seed"


def test_ragged_reference_matches_a_plain_loop(tmp_path):
    draft = FAMILIES["ragged_pool"].draft("small_outliers")
    small = draft.workload_src.replace("S, T, D, LMAX = 65536,", "S, T, D, LMAX = 64,")
    # shrink: 64 segments; T scaled accordingly
    wl_src = small.replace(f"{draft.family_params['T']},", "1200,", 1).replace(
        "torch.float16", "torch.float32"
    )
    (tmp_path / "workload.py").write_text(wl_src)
    (tmp_path / "reference.py").write_text(draft.reference_src)
    wl, ref = (
        _load(tmp_path / "workload.py", "wl_small"),
        _load(tmp_path / "reference.py", "ref_small"),
    )
    v, s, off = wl.make_inputs("main", 3, "cpu")
    got = ref.reference(v, s, off)
    want = torch.stack(
        [
            (torch.softmax(s[a:b], 0)[:, None] * v[a:b]).sum(0)
            for a, b in itertools.pairwise(off.tolist())
        ]
    )
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("regime", list(FAMILIES["algebra"].REGIMES))
def test_algebra_reference_runs_on_scaled_down_shapes(tmp_path, regime):
    d = emit(FAMILIES["algebra"].draft(regime), tmp_path, HW, calibrate=False)
    src = (d / "workload.py").read_text()
    dims = FAMILIES["algebra"].REGIMES[regime]["dims"]
    tiny = [max(1, x // 64) for x in dims]
    (tmp_path / "wl.py").write_text(
        src.replace(f"DIMS = {dims}", f"DIMS = {tiny}").replace("BATCH = 64", "BATCH = 3")
    )
    inputs = _load(tmp_path / "wl.py", f"wl_{regime}").make_inputs("main", 0, "cpu")
    out = _load(d / "reference.py", f"ref_{regime}").reference(*inputs)
    assert out.shape[-2:] == (tiny[0], tiny[4]) and torch.isfinite(out).all()
