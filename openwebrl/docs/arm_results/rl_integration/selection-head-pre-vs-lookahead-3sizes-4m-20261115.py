"""Lookahead vs pre-execution labels pooled over three head sizes (23K linear probe, 4.4M, 22.6M), dropping the 1.8B head
(variant of selection-head-pre-vs-lookahead-3sizes-20261115.py). Same window, runs, bootstrap and records as
selection-head-pre-vs-lookahead-20261115.py, whose code it reuses. Writes the adjacent JSON."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "selection-head-pre-vs-lookahead-20261115.py").read_text()
g = {"__file__": str(HERE / "selection-head-pre-vs-lookahead-20261115.py")}
exec(compile(src[:src.index("for s, name, _ in SIZES:")], "selection-head-pre-vs-lookahead-20261115.py", "exec"), g)
keys = ["probe", "d128", "d512"]
out = {"sizes": keys, "three sizes: pre - lookahead": g["stats"]([f"{k}-pre" for k in keys], [f"{k}-lookahead" for k in keys])}
(HERE / "selection-head-pre-vs-lookahead-3sizes-4m-20261115.json").write_text(json.dumps(out, indent=1) + "\n")
print(out)
