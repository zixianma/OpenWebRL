"""Render the user-supplied six-method overall success comparison."""
import json
from pathlib import Path
import runpy
HERE = Path(__file__).resolve().parent
STEM = "selection-recipe3-vs-cost-sft-actors-overall-baselines-20261113"
if __name__ == "__main__":
    source = json.loads((HERE / "selection-updated-baselines-20261113.json").read_text())
    renderer = runpy.run_path(str(HERE / "selection-recipe3-vs-cost-20261104.py"))
    updates = source["updates"]
    sft = [m for m in renderer["methods"] if not m[0].startswith("rl")]
    for row in updates:
        point = renderer["plot_point"](row["plot_key"], "mean", "overall", circle_encoding=True, updates=updates)
        assert point["success"] == row["reported_mean_overall"]
    print(renderer["render"](("mean",), STEM, 1.42, sft, success_metric="overall",
                             include_rl=False, include_baselines=True, success_updates=updates,
                             method_keys=[row["plot_key"] for row in updates]))
