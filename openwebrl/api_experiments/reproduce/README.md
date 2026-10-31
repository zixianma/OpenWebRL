# Run BrowserCode experiments

Use **[browsercode.py](browsercode.py)** for `setup`, `run`, `check`, `report` and judge-only `rejudge`. The **[BrowserCode guide](../../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart)** contains the complete commands, requirements and methods.

<a id="two-commands"></a>
<a id="what-execution-saves"></a>
<a id="scope-of-reproduction"></a>

- [Setup and evaluation](../../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart)
- [Campaigns and arms](../../docs/API_EXPERIMENT_METHODS.md#main-browsercode-experiments)
- [Saved outputs](../../docs/API_EXPERIMENT_METHODS.md#what-execution-saves)
- [Rejudge saved endpoints with Terra](../../docs/API_EXPERIMENT_METHODS.md#rejudge-saved-endpoints)
- [Scope of reproduction](../../docs/API_EXPERIMENT_METHODS.md#scope-of-reproduction)
- [Offline validation](../../docs/API_EXPERIMENT_METHODS.md#offline-validation)

From the repository root, run `python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py --help` to inspect the interface. Setup prepares dependencies and a frozen runtime. Evaluation selects the correct provider automatically and defaults to a read-only plan; `--execute` requires explicit model and judge spending caps.

<details>
<summary>Lower-level and archived entry points</summary>

The setup/run interface delegates to the existing builders and metered evaluation wrappers. Historical harnesses and their commands remain in the guide's collapsed [archive](../../docs/API_EXPERIMENT_METHODS.md#archived-implementations-and-campaign-records).

</details>
