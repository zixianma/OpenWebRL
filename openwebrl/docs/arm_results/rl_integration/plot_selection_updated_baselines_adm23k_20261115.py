"""Variant of plot_selection_updated_baselines_20261113.py (SFT-actor overall-success cost figure) with the 23K linear-probe head
(ADM-23K) in place of the 22.6M head (ADM-23M). Every other point is unchanged. ADM-23K: system parameters 4.44B + 23,041;
PFLOP and model seconds per task measured with the same accounting (run 1 of the November 14-15 window,
selection-adm23k-cost-20261115.json); success = its raw mean overall success in that window (3 runs, 44.6%), which comes from a
different window than the other points (as the ADM-23M point did)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
STEM = "selection-recipe3-vs-cost-sft-actors-overall-baselines-adm23k-20261115"
if __name__ == "__main__":
    src = (HERE / "selection-recipe3-vs-cost-20261104.py").read_text()
    # legend display text only; internal keys and labels stay the same
    src = src.replace(', "ADM-23M"))', ', "ADM-23K"))').replace('"SFT 4B + ADM-23M")', '"SFT 4B + ADM-23K")')
    g = {"__file__": str(HERE / "selection-recipe3-vs-cost-20261104.py"), "__name__": "renderer"}
    exec(compile(src, "selection-recipe3-vs-cost-20261104.py", "exec"), g)
    probe = json.loads((HERE / "selection-adm23k-cost-20261115.json").read_text())["systems"]["adm-23k"]
    g["EXTERNAL"]["head-gpt55-k3"] = probe
    g["methods"][:] = [(k, l, probe["system_parameters"] if k == "head-gpt55-k3" else p, c, m, h) for k, l, p, c, m, h in g["methods"]]
    updates = json.loads((HERE / "selection-updated-baselines-20261113.json").read_text())["updates"]
    sft = [m for m in g["methods"] if not m[0].startswith("rl")]
    assert g["plot_point"]("head-gpt55-k3", "mean", "overall", circle_encoding=True, updates=updates)["success"] == probe["reported_mean_overall"]
    print(g["render"](("mean",), STEM, 1.42, sft, success_metric="overall", include_rl=False, include_baselines=True, success_updates=updates,
                      method_keys=[*[row["plot_key"] for row in updates], "sft8b", "adm-distilled", "head-gpt55-k3"]))
