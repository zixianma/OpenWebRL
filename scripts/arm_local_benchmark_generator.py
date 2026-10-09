"""Actor-only local WebVoyager/DeepShop evaluation with durable task records.

Copy into a frozen source as ``openwebrl/eval_local_benchmarks.py``. Each
benchmark must run in its own awaited worker and artifact directory. Judge
implementations remain the released benchmark modules, with no rejudging.
"""
from copy import copy
import importlib
import os
from pathlib import Path


SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20,
                max_new_tokens=4096, repetition_penalty=1.0)
BENCHMARKS = ("webvoyager", "deepshop")


def configure(args, sampling_params, benchmark):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported local benchmark: {benchmark!r}")
    # The browser initializer gives the environment override precedence over
    # its YAML. Require both explicitly local instead of silently changing a
    # shared process environment after asynchronous tasks have started.
    if os.environ.get("SLIME_BROWSER_ENV_MODE") != "local_process":
        raise ValueError("Local benchmark requires SLIME_BROWSER_ENV_MODE=local_process")
    for field in ("browser_env", "browser_env_mode"):
        if getattr(args, field, "local_process") != "local_process":
            raise ValueError(f"Local benchmark requires {field}=local_process")
    if getattr(args, "browser_action_selector", None) is not None:
        raise ValueError("Local benchmark requires a standalone actor")
    result = copy(args)
    result.browser_env = "local_process"
    result.browser_env_mode = "local_process"
    result.max_steps = 30
    result.rollout_max_context_len = 32768
    result.context_num_screenshots = 1
    result.turn_history_reasoning_mode = "full"
    result.browser_response_format_mode = "browser_env"
    result.browser_include_tool_response = 1
    result.judge_api_model = "gpt-4o"
    result.judge_prompt_variant = benchmark
    result.judge_max_attached_imgs = 30
    result.rollout_task_timeout_secs = 600.0
    result.task_timeout_secs = 600.0
    result.judge_timeout_secs = 120.0
    result.inference_step_timeout_secs = 120.0
    result.rollout_temperature = SAMPLING["temperature"]
    result.rollout_max_response_len = SAMPLING["max_new_tokens"]
    return result, dict(sampling_params, **SAMPLING)


def validate_local_config(config_path):
    """Check the exact YAML read by generate_browser before any browser call."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict) or config.get("mode") != "local_process":
        raise ValueError("Local benchmark requires frozen env/config.yaml mode: local_process")
    return config


async def _generate(args, sample, sampling_params, benchmark, evaluation):
    if not evaluation:
        raise ValueError("Local benchmark generator is evaluation-only")
    eval_args, sampling = configure(args, sampling_params, benchmark)
    browser = importlib.import_module("openwebrl.generate_browser")
    validate_local_config(Path(browser.__file__).resolve().parent / "env/config.yaml")
    judge = importlib.import_module(f"openwebrl.eval.reward_{benchmark}")
    from openwebrl.eval_monitor import persist_task
    from slime.utils.types import Sample

    turns, error_type = [], None
    try:
        turns = await browser.generate_turn_sample(eval_args, sample, sampling)
        if turns and not any(turn.status == Sample.Status.ABORTED for turn in turns):
            rewards = await judge.reward_func(eval_args, turns)
            for turn, reward in zip(turns, rewards, strict=True):
                # Missing native verdicts remain invalid. A numeric placeholder
                # prevents the generic reward path from silently rejudging them.
                turn.reward = reward if reward is not None else 0.0
                if reward is None:
                    turn.remove_sample = True
                    if turn.metadata is None:
                        turn.metadata = {}
                    turn.metadata["judge_invalid"] = True
                    turn.metadata["benchmark_judge_reward"] = None
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        await persist_task(eval_args, sample, turns, error_type)
    return turns


async def generate_webvoyager(args, sample, sampling_params, evaluation=False):
    return await _generate(args, sample, sampling_params, "webvoyager", evaluation)


async def generate_deepshop(args, sample, sampling_params, evaluation=False):
    return await _generate(args, sample, sampling_params, "deepshop", evaluation)
