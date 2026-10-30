# API experiments

The **[API experiments guide](../docs/API_EXPERIMENT_METHODS.md)** is the main reference for the studies, methods, runnable commands, implementation and result evidence.

<a id="start-here"></a>
<a id="restore-an-original-campaign-layout"></a>

| To… | Follow… |
|---|---|
| Understand the studies and their differences | [Experiments and protocols](../docs/API_EXPERIMENT_METHODS.md#experiments) |
| Run Luna or Claude BrowserCode episodes | [Reproduction quickstart](../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart) |
| Trace workers, models, budgets, browsers and judges | [Campaign code map](../docs/API_EXPERIMENT_METHODS.md#code-and-frozen-artifacts) |
| Restore original source filenames | [Source reconstruction](../docs/API_EXPERIMENT_METHODS.md#restore-an-original-campaign-layout) · [provenance index](source-provenance.json) |
| Check the public code offline | [Validation commands and test scope](../docs/API_EXPERIMENT_METHODS.md#offline-validation) |
| Find results and qualified evidence | [Results and analysis](../docs/API_EXPERIMENT_METHODS.md#results-and-analysis-pointers) |

<a id="layout"></a>

`browsercode/` and `cuawright/` preserve frozen source snapshots, grouped by harness and function. They retain original imports and runtime paths. [materialize.py](materialize.py) restores the original layout; [reproduce/](reproduce/) provides Luna/Claude wrappers into an existing configured runtime. Private task manifests and browser dependencies remain local. The guide explains these requirements and the distinction between reconstructing code and executing episodes.
