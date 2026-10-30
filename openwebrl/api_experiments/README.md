# API experiments

The **[API experiments guide](../docs/API_EXPERIMENT_METHODS.md)** is the main reference for the studies, methods, runnable commands, implementation and result evidence.

<a id="start-here"></a>
<a id="restore-an-original-campaign-layout"></a>

| To… | Follow… |
|---|---|
| Understand the studies and their differences | [Experiments and protocols](../docs/API_EXPERIMENT_METHODS.md#experiments) |
| Install and run the experiments | [Reproduction quickstart](../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart) |
| Trace workers, models, budgets, browsers and judges | [Campaign code map](../docs/API_EXPERIMENT_METHODS.md#code-and-frozen-artifacts) |
| Restore original source filenames | [Source reconstruction](../docs/API_EXPERIMENT_METHODS.md#restore-an-original-campaign-layout) · [provenance index](source-provenance.json) |
| Check the public code offline | [Validation commands and test scope](../docs/API_EXPERIMENT_METHODS.md#offline-validation) |
| Find results and qualified evidence | [Results and analysis](../docs/API_EXPERIMENT_METHODS.md#results-and-analysis-pointers) |

<a id="layout"></a>

`browsercode/` and `cuawright/` preserve frozen source snapshots. [portable_assets/](portable_assets/) supplies scientific configs, prompts, support code and benchmark-ID rosters. [reproduce/](reproduce/) installs pinned dependencies, builds isolated runtimes and runs BrowserCode, native CUAWright and earlier coordinate-browser experiments. Start with the guide’s [clean setup commands](../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart); credentials and run outputs stay local.
