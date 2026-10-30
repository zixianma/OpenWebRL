# Reproduce the BrowserCode API evaluations

The commands and runtime requirements now live in the **[API experiments guide](../../docs/API_EXPERIMENT_METHODS.md#reproduction-quickstart)** alongside the study methods, code map and results.

<a id="two-commands"></a>
<a id="what-execution-saves"></a>
<a id="scope-of-reproduction"></a>

- [Clean installation and runtime preparation](../../docs/API_EXPERIMENT_METHODS.md#install-the-pinned-dependencies)
- [BrowserCode campaigns and arms](../../docs/API_EXPERIMENT_METHODS.md#main-browsercode-experiments)
- [Native CUAWright commands](../../docs/API_EXPERIMENT_METHODS.md#native-cuawright-experiments)
- [Dry-run commands](../../docs/API_EXPERIMENT_METHODS.md#dry-run-commands)
- [Saved outputs and accounting](../../docs/API_EXPERIMENT_METHODS.md#what-execution-saves)
- [Scope and limits of reproduction](../../docs/API_EXPERIMENT_METHODS.md#scope-of-reproduction)
- [Offline validation](../../docs/API_EXPERIMENT_METHODS.md#offline-validation)

[install.py](install.py) prepares pinned dependencies; [prepare_browsercode.py](prepare_browsercode.py) builds a runtime from the published scientific inputs. [eval_luna.py](eval_luna.py), [eval_claude.py](eval_claude.py), [eval_gpt.py](eval_gpt.py), [native.py](native.py) and [legacy.py](legacy.py) provide run entry points. Episode commands default to read-only dry runs; `--execute` uses explicit run caps. Use `--help` to inspect options without a runtime.
