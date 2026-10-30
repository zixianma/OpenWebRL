"""Passive cumulative timings; nested phases overlap and must not be summed."""
import functools
import time


def phase_timed(name):
    def decorate(function):
        @functools.wraps(function)
        async def wrapped(self, *args, **kwargs):
            started = time.monotonic()
            failed = True
            try:
                result = await function(self, *args, **kwargs)
                failed = False
                return result
            finally:
                phases = self.__dict__.setdefault('browser_phase_metrics', {})
                row = phases.setdefault(name, {'calls': 0, 'raised': 0, 'seconds': 0.0})
                row['calls'] += 1
                row['raised'] += int(failed)
                row['seconds'] += time.monotonic() - started
        return wrapped
    return decorate
