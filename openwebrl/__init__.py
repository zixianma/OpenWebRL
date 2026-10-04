"""OpenWebRL browser agents, with lazy runtime imports for offline data tooling."""
from importlib import import_module

_EXPORTS = {
    **{name: ('openwebrl.base.types', name) for name in (
        'Status', 'InteractionResult', 'EnvConfig', 'ToolCall', 'ParsedToolResult')},
    'BaseGymEnvAdapter': ('openwebrl.base.adapter', 'BaseGymEnvAdapter'),
    'EnvRegistry': ('openwebrl.base.registry', 'EnvRegistry'),
    'ENV_REGISTRY': ('openwebrl.base.registry', 'DEFAULT_REGISTRY'),
    **{name: ('openwebrl.base.utils', name) for name in (
        'ToolParser', 'TokenHandler', 'tools_to_openai_format', 'calls_to_action', 'TOOL_INSTRUCTION')},
    **{name: ('openwebrl.generate_browser', name) for name in (
        'generate_trajectory_sample', 'generate_turn_sample')},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
