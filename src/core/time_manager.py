"""Routines to time events and restrict logs or operations by frequency."""
import time

import numpy as np

import logging
logger = logging.getLogger(__name__)


class TimeManager(object):
    """Manage timing of event executions or measure timings."""

    def __init__(self, model):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model

        self._timers = {}
        self._last_time = {}

    def start(self, name, **kwargs):
        """Begin timer for given event/operation."""
        if name not in self._timers:
            timer = Timer(**kwargs)
            self._timers[name] = timer
        else:
            timer = self._timers[name]
        timer.start()

    def end(self, name):
        """End timer for given event/operation."""
        assert name in self._timers
        return self._timers[name].end()

    def has_been_n_seconds_since_last(self, identifier, seconds):
        """Indicate if enough time has passed since last time.

        Also updates the `last time` record based on identifier.
        """
        current_time = time.time()
        if identifier not in self._last_time or \
           (current_time - self._last_time[identifier] > seconds):
            self._last_time[identifier] = current_time
            return True
        return False

    def log_every(self, identifier, message, seconds=1):
        """Limit logging of messages based on specified interval and identifier."""
        if self.has_been_n_seconds_since_last(identifier, seconds):
            logger.info(message)
        else:
            logger.debug(message)


# A local Timer class for timing
class Timer(object):
    """Record start and end times as requested and provide summaries."""

    def __init__(self, average_over_last_n_timings=10):
        """Store keyword parameters."""
        self._average_over_last_n_timings = average_over_last_n_timings
        self._active = False
        self._timings = []
        self._start_time = -1

    def start(self):
        """Cache starting time."""
        # assert not self._active
        self._start_time = time.time()
        self._active = True

    def end(self):
        """Check ending time and store difference."""
        assert self._active and self._start_time > 0

        # Calculate difference
        end_time = time.time()
        time_difference = end_time - self._start_time

        # Record timing (and trim history)
        self._timings.append(time_difference)
        if len(self._timings) > self._average_over_last_n_timings:
            self._timings = self._timings[-self._average_over_last_n_timings:]

        # Reset
        self._start_time = -1
        self._active = False

        return time_difference

    @property
    def current_mean(self):
        """Calculate mean timing for as many trials as specified in constructor."""
        values = self._timings
        return np.mean(values)
