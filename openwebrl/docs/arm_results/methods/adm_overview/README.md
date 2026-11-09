# Action Decision Models overview (paper Figure 1)

![Action Decision Models overview](figure1.png)

[PDF](figure1.pdf) · [SVG](figure1.svg) · [Plot script](plot_figure1.py)

Counterclockwise exports: [PNG](figure1-ccw.png) · [PDF](figure1-ccw.pdf) · [SVG](figure1-ccw.svg)

## Reproduce

From the repository root, install `requirements.txt` in your Python environment, then run:

```sh
python -m pip install -r openwebrl/docs/arm_results/methods/adm_overview/requirements.txt
python openwebrl/docs/arm_results/methods/adm_overview/plot_figure1.py
```

The script writes PNG, PDF and SVG beside itself, plus counterclockwise exports.
It imports the repository's `openwebrl/docs/paper_plot_style.py` and bundled Albert Sans fonts.
No browser session or experiment service is required.

## Sources

The layout and numerical results retain the paper's Figure 1 snapshot from paper commit `6f3f7c9`. The diagram uses yellow with amber borders for browser states, salmon for candidate and selected actions, salmon for lookahead evidence and the teacher, and light purple for ADM and its reward signal. State/action token colors identify content, not terminal outcomes. Actor policies stay neutral, and the result insets keep purple ADM and gray control series. Lighter purple fills denote secondary ADM bars. The script imports `paper_plot_palette.py` alongside the shared typography; no data, smoothing, screenshot content or geometry is changed.

- The real browser observation is the Dinka Wikipedia task state shown in the paper. `web_states/dinka/state.jpg` is the original observation, resized without cropping. Its SHA-256 is `d028c34171bf1fd94709663c7c071f090cbce4bab4751db77d3ab8df8a4080be`. Lookahead icons, teacher choice and model scores are schematic.
- Inference inset: November 4 SFT-only, three-run valid-only benchmark means from `selection-recipe3-vs-cost-20261104.json` (Git blob `5fcc4d9819540dde6285174759a82d15f85ef301`), embedded in the script. The plotted coordinates match the paper's Figure 4 snapshot.
- Offline inset: the three aggregate `valid_success` values used by the paper's `preliminary_results.json`, retained in the local file of the same name.
- Online inset: outcome-only and Gate B aggregate checkpoint counts through iteration 90 and the original smoothing bandwidth from the paper's `results_20261025.json`, retained in the local file of the same name.

The three application insets summarize separate studies, rather than a matched cross-application comparison. These JSON files contain only aggregates used by this figure. The bundle contains no rollout trajectories or task-pool dataset.
