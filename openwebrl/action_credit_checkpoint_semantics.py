"""Absolute scheduler restoration at the source-owned native load boundary.

The pinned Megatron loader assumes a freshly initialized scheduler: its
load_state_dict increments num_steps. Action-credit also restores into live
instances. This argument proxy changes only actual scheduler-load calls;
skipped optimizer/scheduler loads do not touch the underlying object. It does
not change the installed package or mutate its global class/methods.
"""
from collections.abc import Mapping

class AbsoluteSchedulerLoad:
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError('Native scheduler checkpoint must be a mapping')
        expected = state.get('num_iters', state.get('num_steps'))
        previous = self._scheduler.num_steps
        if type(expected) is not int or expected < 0 or type(previous) is not int or previous < 0:
            raise ValueError('Native scheduler requires nonnegative integer saved/live counters')
        self._scheduler.num_steps = 0
        try:
            result = self._scheduler.load_state_dict(state)
            if self._scheduler.num_steps != expected:
                raise ValueError('Installed scheduler did not restore the exact saved counter')
            return result
        except BaseException as error:
            self._scheduler.num_steps = previous
            error.add_note('Native checkpoint restore failed; other model/optimizer/scheduler fields may be partially loaded; do not continue training')
            raise

def native_checkpoint_scheduler(scheduler):
    return None if scheduler is None else AbsoluteSchedulerLoad(scheduler)
