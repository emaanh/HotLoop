"""hotloop-taskgen list | emit --family F --regime R --out DIR | emit-all --out DIR"""

from __future__ import annotations

import argparse
from pathlib import Path

from hotloop_taskgen.core import emit
from hotloop_taskgen.families import FAMILIES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-taskgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("emit", "emit-all"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", required=True, type=Path)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--gpu-sku", default="A100-SXM4-40GB")
        sp.add_argument("--image", default="unpinned")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--no-calibrate", action="store_true")
        if name == "emit":
            sp.add_argument("--family", required=True, choices=sorted(FAMILIES))
            sp.add_argument("--regime", required=True)
    a = p.parse_args(argv)

    if a.cmd == "list":
        for fam, mod in FAMILIES.items():
            for regime, params in mod.REGIMES.items():
                print(
                    f"{fam:14s} {regime:32s} {'control' if params.get('control') else 'headroom'}"
                )
        return 0

    todo = (
        [(a.family, a.regime)]
        if a.cmd == "emit"
        else [(f, r) for f, mod in FAMILIES.items() for r in mod.REGIMES]
    )
    hardware = {"gpu_sku": a.gpu_sku, "image": a.image}
    for fam, regime in todo:
        d = emit(
            FAMILIES[fam].draft(regime, a.seed),
            a.out,
            hardware,
            calibrate=not a.no_calibrate,
            device=a.device,
        )
        print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
