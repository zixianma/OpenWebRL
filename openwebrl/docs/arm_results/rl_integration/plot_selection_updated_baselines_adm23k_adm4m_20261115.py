"""Variant of plot_selection_updated_baselines_20261113.py (SFT-actor overall-success cost figure) that adds the 23K linear-probe
head (ADM-23K) and the 4.4M d128-L2 head (ADM-4M, training seed 0) next to ADM-23M and ADM-4B; every existing point is unchanged.
ADM-23K: system parameters 4.44B + 23K, cost and success from selection-adm23k-cost-20261115.json (same accounting, raw
November 14-15 success 44.6%). ADM-4M: system parameters 4.44B + 4.36M;
PFLOP and model seconds per task measured with the same accounting (run 1 of the November 14-15 window,
selection-adm4m-cost-20261115.json); success = its raw mean overall success in that window (3 runs, 45.7%), a different window
from the other points (as the ADM-23M point is). Over three training seeds the 4.4M and 22.6M heads are level (-0.2 [-1.8, +1.4])."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
STEM = "selection-recipe3-vs-cost-sft-actors-overall-baselines-adm23k-adm4m-20261115"
KEY = "head-4m-gpt55"
KEY23K = "head-23k-gpt55"
if __name__ == "__main__":
    src = (HERE / "selection-recipe3-vs-cost-20261104.py").read_text()
    edits = [
        ('    ("head-gpt55-k3", "ADM-22M (GPT-5.5)", HEAD, C["ADM_22M"], "o", False),\n',
         '    ("head-gpt55-k3", "ADM-22M (GPT-5.5)", HEAD, C["ADM_22M"], "o", False),\n    ("head-23k-gpt55", "ADM-23K (GPT-5.5)", 4.44e9 + 23041, C["ADM_22M"], "o", False),\n    ("head-4m-gpt55", "ADM-4M (GPT-5.5)", 4.44e9 + 4.36e6, C["ADM_22M"], "o", False),\n'),
        ('    "head-gpt55-k3": ("#B69AE8", "D"),\n', '    "head-gpt55-k3": ("#B69AE8", "D"),\n    "head-23k-gpt55": ("#E6DDFA", "D"),\n    "head-4m-gpt55": ("#CFBDF6", "D"),\n'),
        ('size = 44 if key in ("head-gpt55-k3",', 'size = 44 if key in ("head-23k-gpt55", "head-4m-gpt55", "head-gpt55-k3",'),
        # thin rings mark frontier points, so the parameters-panel frontier (two points at ~4.44B) stays visible
        ('                  frontier_rings=draw_rings, updates=updates,', '                  frontier_rings=True, updates=updates,'),
        ('                own_entries.append((by_label["ADM-23M (GPT-5.5)"], "ADM-23M"))\n',
         '                if "ADM-23K (GPT-5.5)" in by_label:\n                    own_entries.append((by_label["ADM-23K (GPT-5.5)"], "ADM-23K"))\n                if "ADM-4M (GPT-5.5)" in by_label:\n                    own_entries.append((by_label["ADM-4M (GPT-5.5)"], "ADM-4M"))\n                own_entries.append((by_label["ADM-23M (GPT-5.5)"], "ADM-23M"))\n'),
    ]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    g = {"__file__": str(HERE / "selection-recipe3-vs-cost-20261104.py"), "__name__": "renderer"}
    exec(compile(src, "selection-recipe3-vs-cost-20261104.py", "exec"), g)
    g["EXTERNAL"][KEY23K] = json.loads((HERE / "selection-adm23k-cost-20261115.json").read_text())["systems"]["adm-23k"]
    g["EXTERNAL"][KEY] = json.loads((HERE / "selection-adm4m-cost-20261115.json").read_text())["systems"]["adm-4m"]
    updates = json.loads((HERE / "selection-updated-baselines-20261113.json").read_text())["updates"]
    sft = [m for m in g["methods"] if not m[0].startswith("rl")]
    assert g["plot_point"](KEY, "mean", "overall", circle_encoding=True, updates=updates)["success"] == 45.7
    print(g["render"](("mean",), STEM, 1.42, sft, success_metric="overall", include_rl=False, include_baselines=True, success_updates=updates,
                      method_keys=[*[row["plot_key"] for row in updates], "sft8b", "adm-distilled", "head-gpt55-k3", KEY23K, KEY]))
