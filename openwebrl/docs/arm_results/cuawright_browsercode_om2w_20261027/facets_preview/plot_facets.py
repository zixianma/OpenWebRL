"""Render the separate-harness layout at its established preview URL."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("harness_tradeoffs", ROOT / "plot_om2w_harness_tradeoffs.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
plot.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
_, browsercode, native = plot.read_rows(ROOT / "om2w_harness_comparison_valid_only.json")
plot.figure(browsercode, native, plot.METRICS, Path(__file__).resolve().parent, "om2w_harness_facets")
