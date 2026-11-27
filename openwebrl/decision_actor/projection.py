"""Map an actor tool call onto the closed option set.

Used two ways: (1) offline coverage of saved trajectories, and (2) the live
``projected_actor`` decider, which executes the SFT actor's own choice snapped to
the nearest option. (2) isolates what the action space and harness lose, with no
learned decision model involved: if projected-SFT matches unconstrained SFT, the
action space is not the bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from openwebrl.decision_actor.options import Option

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_call(text: str) -> dict | None:
    match = TOOL_CALL.search(text or "")
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    call.setdefault("arguments", {})
    return call


@dataclass
class Projection:
    option_id: str | None   # None: not representable in this action space
    reason: str             # how it matched, or why not
    distance: float = 0.0   # click distance to the matched element (0 when inside its box), in 0..1 units


def _inside(point, box, pad=0.0):
    return box[0] - pad <= point[0] <= box[2] + pad and box[1] - pad <= point[1] <= box[3] + pad


def _dist(point, box):
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return (dx * dx + dy * dy) ** .5


def project(call: dict | None, options: list[Option], *, radius: float = 0.03, pending_write: bool = False) -> Projection:
    """Snap one tool call (OpenWebRL schema, point_2d in 0..1000) to an option id."""
    if not call:
        return Projection(None, "unparsable")
    name, args = call["name"], call.get("arguments") or {}
    by_id = {o.id: o for o in options}
    if name in ("click", "hover"):
        point = args.get("point_2d")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            return Projection(None, "bad_point")
        p = (float(point[0]) / 1000, float(point[1]) / 1000)
        kinds = ("hover",) if name == "hover" else ("click", "type")
        if args.get("clicks", 1) not in (1, None) or args.get("button", "left") != "left":
            return Projection(None, "multi_or_right_click")
        candidates = [o for o in options if o.element and o.kind in kinds]
        inside = [o for o in candidates if _inside(p, o.element["bbox"])]
        if inside:
            area = lambda o: (o.element["bbox"][2] - o.element["bbox"][0]) * (o.element["bbox"][3] - o.element["bbox"][1])
            best_area = min(area(o) for o in inside)
            smallest = [o for o in inside if area(o) == best_area]
            # A click on a typable field is a "type" option only if the actor types next; prefer the plain click.
            chosen = next((o for o in smallest if o.kind != "type"), smallest[0])
            return Projection(chosen.id, "inside_element")
        near = sorted(((_dist(p, o.element["bbox"]), o) for o in candidates if o.kind != "type"), key=lambda x: x[0])
        if near and near[0][0] <= radius:
            return Projection(near[0][1].id, "near_element", near[0][0])
        grid = [o for o in options if o.kind == "grid"]
        if grid and name == "click":
            best = min(grid, key=lambda o: _dist(p, [v / 1000 for v in o.action["arguments"]["point_2d"]] * 2))
            return Projection(best.id, "grid")
        return Projection(None, "no_element_at_point", near[0][0] if near else 1.0)
    if name == "write":
        return Projection("__typed__" if pending_write else None, "write_after_type" if pending_write else "write_without_type_option")
    if name == "scroll":
        direction = args.get("direction", "down")
        target = {"down": "scroll_down", "up": "scroll_up"}.get(direction)
        if target in by_id:
            return Projection(target, "scroll_page" if not args.get("point_2d") else "scroll_container_as_page")
        return Projection(None, f"scroll_{direction}")
    if name == "press_keys":
        keys = [str(k).lower() for k in (args.get("keys") or [])] if not isinstance(args.get("keys"), str) else [args["keys"].lower()]
        for key, target in (("enter", "press_enter"), ("return", "press_enter"), ("escape", "press_escape"), ("esc", "press_escape")):
            if keys == [key] and target in by_id:
                return Projection(target, "key")
        return Projection(None, "other_keys:" + "+".join(keys))
    simple = dict(go_back="go_back", wait="wait", done="done")
    if name in simple:
        return Projection(simple[name] if simple[name] in by_id else None, name)
    if name == "switch_tab":
        target = f"tab{args.get('tab_index')}"
        return Projection(target if target in by_id else None, "tab")
    return Projection(None, "unsupported:" + name)
