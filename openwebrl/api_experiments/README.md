# BrowserCode API experiments

Start with the **[BrowserCode guide](../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart)**: one setup command prepares the dependencies and frozen runtime; one evaluation command runs the selected arm with explicit caps.

<a id="start-here"></a>

| To… | Follow… |
|---|---|
| Set up and run | [Quickstart](../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart) · [browsercode.py](reproduce/browsercode.py) |
| Choose models and methods | [Campaigns and arms](../docs/API_EXPERIMENT_METHODS.md#main-browsercode-experiments) |
| Understand the protocol | [Experiments](../docs/API_EXPERIMENT_METHODS.md#experiments) · [metrics](../docs/API_EXPERIMENT_METHODS.md#metrics-and-comparability) |
| Trace the implementation | [Code map](../docs/API_EXPERIMENT_METHODS.md#code-and-frozen-artifacts) · [offline checks](../docs/API_EXPERIMENT_METHODS.md#offline-validation) |
| Find qualified results | [Results and analysis](../docs/API_EXPERIMENT_METHODS.md#results-and-analysis-pointers) |

<a id="layout"></a>

[browsercode/](browsercode/) holds frozen source snapshots; [portable_assets/](portable_assets/) supplies configurations, prompts, support code and fixed task-ID rosters. [reproduce/browsercode.py](reproduce/browsercode.py) provides `setup`, `run`, `check` and `report`. Evaluation defaults to a read-only plan; add `--execute` to use the required run caps. Credentials and raw outputs stay local.

<details>
<summary>Archived implementations and source provenance</summary>

<a id="restore-an-original-campaign-layout"></a>

Earlier harnesses remain preserved for historical reproduction. See the guide's collapsed [archive](../docs/API_EXPERIMENT_METHODS.md#archived-implementations-and-campaign-records), [source reconstruction](../docs/API_EXPERIMENT_METHODS.md#restore-an-original-campaign-layout) and [provenance index](source-provenance.json).

</details>
