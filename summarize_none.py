# summarize_none.py
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for f in sorted(root.rglob("targets/none/target_meta.json")):
    meta = json.load(open(f))
    it = meta["iterations"][0]
    info = it["measurement"]
    rows.append((f.parents[2].name, meta["m"], it["w_hat"], info.get("alpha"),
                 info.get("a0"), info.get("relative_residual"), it["max_abs_error"]))

fmt = lambda xs: "[" + ", ".join(f"{x:.2f}" for x in xs) + "]" if xs else "-"
for name, m, w_hat, alpha, a0, res, err in rows:
    print(f"{name[:40]:40s}  m={fmt(m)}  w_hat={fmt(w_hat)}  alpha={fmt(alpha)}  "
          f"a0={a0 if a0 is None else f'{a0:.2f}'}  resid={res if res is None else f'{res:.2f}'}  err={err:.2f}")