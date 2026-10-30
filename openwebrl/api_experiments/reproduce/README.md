# Reproduce the BrowserCode API evaluations

The commands and runtime requirements now live in the **[API experiments guide](../../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart)** alongside the study methods, code map and results.

<a id="two-commands"></a>
<a id="what-execution-saves"></a>
<a id="scope-of-reproduction"></a>

- [Dry-run commands and supported arms](../../docs/API_EXPERIMENT_METHODS.md#dry-run-commands)
- [Saved outputs and accounting](../../docs/API_EXPERIMENT_METHODS.md#what-execution-saves)
- [Scope and limits of reproduction](../../docs/API_EXPERIMENT_METHODS.md#scope-of-reproduction)
- [Offline validation](../../docs/API_EXPERIMENT_METHODS.md#offline-validation)

[eval_luna.py](eval_luna.py) and [eval_claude.py](eval_claude.py) default to a read-only dry run. They require an existing configured frozen campaign runtime; `--execute` starts browser/API work within explicit run caps. Use `--help` to inspect options without a runtime.
