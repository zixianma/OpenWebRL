"""Named colors shared by paper diagrams and quantitative result plots.

STATE is yellow, ACTION is mint, ADM is purple and lookahead teachers are salmon.
Neutral controls, model-size shades and source-label outlines retain explicit roles.
Color changes never affect source measurements, intervals or data populations.
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "assets/paper_plot_palette.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
COLORS = dict(CONFIG["trial"]["colors"])


def tint(role, white_fraction):
    """Blend a named role toward white for a diagram/background fill."""
    if not 0 <= white_fraction <= 1:
        raise ValueError("white_fraction must be between 0 and 1")
    hex_color = COLORS[role].lstrip("#")
    channels = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(value * (1 - white_fraction) + 255 * white_fraction):02X}"
                          for value in channels)
