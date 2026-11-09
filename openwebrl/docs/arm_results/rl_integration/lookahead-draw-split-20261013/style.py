"""Shared names, colours and panels for the draw-split figures."""
from paper_plot_palette import COLORS as C

STYLE = {  # key in curve.json -> (legend label, colour, line style)
    "Best on observed draws": ("Oracle (observed draws)", C["ORACLE"], "-"),
    "Lookahead K=3": ("Lookahead H=4", C["LOOKAHEAD_TEACHER"], "-"),
    "Lookahead K=2": ("Lookahead H=3", C["LOOKAHEAD_TEACHER"], "--"),
    "Lookahead K=1": ("Lookahead H=2", C["LOOKAHEAD_TEACHER"], ":"),
    "Before execution": ("Pre-execution", C["PRE_EXEC"], "-"),
    "WebArbiter formulation": ("WebArbiter (Luna)", C["BASELINE_DARK"], "-."),
    "WebShepherd formulation": ("WebShepherd (Luna)", C["BASELINE_LIGHT"], "-."),
    "PRO-CUA formulation": ("PRO-CUA (Luna)", C["SINGLE_CALL_MAIN"], "-."),
    "Actor first sample": ("Actor (first sample)", C["BASELINE_DARK"], ":"),
    "Uniform": ("Uniform", C["BASELINE_LIGHT"], "--"),
}
PANELS = [("pooled", "All"), ("lookahead-fast1k-20261013", "New states"), ("lookahead-v2-20261013", "Earlier states")]
TITLES = {"invalid_as_failure": "Invalid draws = failure", "valid_only": "Valid draws only"}
