"""Closed action set for the decision actor: every option is an executable OpenWebRL tool call.

Options come only from the browser observation (``a11ytree`` elements from
``find_all_clickable_elements`` with 0..1 viewport bboxes, open tabs) plus fixed
page actions. Text is never chosen here: a ``type`` option clicks the field and
asks the separate typing model for the message on the next turn, and ``done``
asks the typing model for the final response. All knobs that change the action
space live in :class:`HarnessConfig` so harness variants are explicit and logged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

TEXT_FIELDS = ("innerText", "ariaLabel", "placeholder", "value", "title", "alt", "textContent",
               "dataTooltip", "ariaDescription", "dataDescription")
TYPABLE_INPUT_TYPES = {None, "", "text", "search", "email", "url", "tel", "number", "password", "date"}
PAGE_ACTIONS = ("scroll_down", "scroll_up", "go_back", "wait", "press_enter", "press_escape", "done")


@dataclass(frozen=True)
class HarnessConfig:
    """Every action-space / harness knob. Change one at a time and compare against the projection ceiling."""
    max_elements: int = 120            # keep the first N elements in reading order
    dedupe_iou: float = 0.9            # merge near-identical boxes (keeps the smaller / more specific one)
    min_box: float = 0.003             # drop boxes thinner than this fraction of the viewport
    label_chars: int = 80              # truncate element text in option labels
    label_bbox: bool = True            # append the element centre (0..1) to its label
    hover_options: bool = False        # add a hover option per element
    type_options: bool = True          # add "type into" options for typable fields
    page_actions: tuple = PAGE_ACTIONS
    scroll_amount: float = 0.5
    wait_seconds: int = 3
    tab_options: bool = True           # switch to each inactive tab
    grid: tuple = (0, 0)               # (cols, rows) click grid for targets without DOM elements; (0, 0) = off
    submit_after_type: str = "decide"  # decide | always: queue press Enter after the typed message
    history_actions: int = 8           # executed option labels kept for the decision state
    page_text_chars: int = 0           # visible page text included in the decision state (0 = none)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict | None) -> "HarnessConfig":
        data = dict(data or {})
        for key in ("page_actions", "grid"):
            if key in data:
                data[key] = tuple(data[key])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown harness knobs: {sorted(unknown)}")
        return cls(**data)


@dataclass
class Option:
    id: str
    kind: str                          # click | hover | type | scroll | back | wait | press | done | tab | grid
    label: str                         # text shown to the decision model
    action: dict | None                # tool call executed now (None for done: typing model writes it)
    element: dict | None = None        # source element (bbox 0..1) when element-based
    needs_text: str | None = None      # "write" (next turn) or "done" (this turn) when the typing model is needed
    extra: dict = field(default_factory=dict)


def element_text(element: dict, limit: int) -> str:
    for key in TEXT_FIELDS:
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text if len(text) <= limit else text[: limit - 1] + "…"
    return ""


def is_typable(element: dict) -> bool:
    tag = (element.get("tag") or "").lower()
    if tag == "textarea":
        return True
    if tag == "input":
        return (element.get("type") or "").lower() in TYPABLE_INPUT_TYPES
    return False


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b):
    inter = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def clean_elements(elements: list[dict], config: HarnessConfig) -> list[dict]:
    """Valid, deduplicated elements in reading order (top-to-bottom, then left-to-right)."""
    kept = []
    for element in elements or []:
        box = element.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        box = [float(v) for v in box]
        if box[2] - box[0] < config.min_box or box[3] - box[1] < config.min_box:
            continue
        if box[2] <= 0 or box[3] <= 0 or box[0] >= 1 or box[1] >= 1:
            continue
        kept.append(dict(element, bbox=box))
    kept.sort(key=lambda e: (round(e["bbox"][1], 2), e["bbox"][0]))
    result = []
    for element in kept:
        twin = next((r for r in result if _iou(r["bbox"], element["bbox"]) >= config.dedupe_iou), None)
        if twin is None:
            result.append(element)
        elif _area(element["bbox"]) < _area(twin["bbox"]) or (not element_text(twin, 1) and element_text(element, 1)):
            result[result.index(twin)] = element
    return result[: config.max_elements]


def centre_1000(box) -> list[int]:
    return [round((box[0] + box[2]) / 2 * 1000), round((box[1] + box[3]) / 2 * 1000)]


def build_options(observation: dict, config: HarnessConfig, *, extra_options: list[Option] | None = None) -> list[Option]:
    """All options for one browser state; ids are stable within the state ("e12", "t12", "scroll_down", ...)."""
    options: list[Option] = []
    elements = clean_elements(observation.get("a11ytree") or [], config)
    for i, element in enumerate(elements):
        tag = (element.get("tag") or "element").lower()
        text = element_text(element, config.label_chars)
        where = ""
        if config.label_bbox:
            c = centre_1000(element["bbox"])
            where = f" at ({c[0] / 1000:.2f}, {c[1] / 1000:.2f})"
        desc = f'[{tag}{"/" + element["type"] if element.get("type") else ""}] "{text}"{where}'
        point = centre_1000(element["bbox"])
        if config.type_options and is_typable(element):
            options.append(Option(f"t{i}", "type", "type into " + desc,
                                  dict(name="click", arguments=dict(point_2d=point)), element, needs_text="write"))
        options.append(Option(f"e{i}", "click", "click " + desc, dict(name="click", arguments=dict(point_2d=point)), element))
        if config.hover_options:
            options.append(Option(f"h{i}", "hover", "hover " + desc, dict(name="hover", arguments=dict(point_2d=point)), element))
    cols, rows = config.grid
    for r in range(rows):
        for c in range(cols):
            point = [round((c + .5) / cols * 1000), round((r + .5) / rows * 1000)]
            options.append(Option(f"g{r}_{c}", "grid", f"click screen region row {r + 1}/{rows}, column {c + 1}/{cols}",
                                  dict(name="click", arguments=dict(point_2d=point))))
    if config.tab_options:
        for tab in observation.get("all_tab_url") or []:
            if not tab.get("active"):
                options.append(Option(f"tab{tab['index']}", "tab", f'switch to tab {tab["index"]}: "{tab.get("title", "")}" {tab.get("url", "")}',
                                      dict(name="switch_tab", arguments=dict(tab_index=tab["index"]))))
    fixed = dict(
        scroll_down=Option("scroll_down", "scroll", "scroll the page down",
                           dict(name="scroll", arguments=dict(direction="down", amount=config.scroll_amount))),
        scroll_up=Option("scroll_up", "scroll", "scroll the page up",
                         dict(name="scroll", arguments=dict(direction="up", amount=config.scroll_amount))),
        go_back=Option("go_back", "back", "go back to the previous page", dict(name="go_back", arguments={})),
        wait=Option("wait", "wait", "wait for the page to load", dict(name="wait", arguments=dict(seconds=config.wait_seconds))),
        press_enter=Option("press_enter", "press", "press Enter (submit the focused field)", dict(name="press_keys", arguments=dict(keys=["Enter"]))),
        press_escape=Option("press_escape", "press", "press Escape (close a popup or menu)", dict(name="press_keys", arguments=dict(keys=["Escape"]))),
        done=Option("done", "done", "stop: the task is complete (give the final answer)", None, needs_text="done"),
    )
    for name in config.page_actions:
        if name not in fixed:
            raise ValueError(f"Unknown page action {name!r}")
        options.append(fixed[name])
    options.extend(extra_options or [])
    ids = [o.id for o in options]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate option ids")
    return options


def describe_action(action: dict | None) -> str:
    if not action:
        return "none"
    return f'{action["name"]}({json.dumps(action.get("arguments", {}), ensure_ascii=False, sort_keys=True)})'


def option_summary(option: Option) -> dict[str, Any]:
    return dict(id=option.id, kind=option.kind, label=option.label, action=option.action, needs_text=option.needs_text)
