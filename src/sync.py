"""
sync.py
-------
Helpers for the synchronous tick loop:

* :func:`make_tick_save_planner` decides, for each world tick, whether the
  current frame should be persisted to disk based on the ratio between
  ``simulation_hz`` and ``save_hz``.
* :class:`FrameCounter` tracks a couple of running stats (saved, dropped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


LOGGER = logging.getLogger(__name__)


def make_tick_save_planner(simulation_hz: float, save_hz: float) -> "TickSavePlanner":
    if simulation_hz <= 0 or save_hz <= 0:
        raise ValueError("simulation_hz and save_hz must be > 0")
    if save_hz > simulation_hz:
        LOGGER.warning(
            "save_hz (%.2f) > simulation_hz (%.2f); will save every tick.",
            save_hz, simulation_hz,
        )
    return TickSavePlanner(simulation_hz=float(simulation_hz), save_hz=float(save_hz))


@dataclass
class TickSavePlanner:
    """Decide which world ticks become *saved* frames.

    We accumulate phase and trigger a save whenever the accumulator crosses
    a (1 / save_hz) boundary. This handles non-integer ratios cleanly
    (e.g. simulation_hz=20, save_hz=7).
    """

    simulation_hz: float
    save_hz: float
    _phase: float = 0.0

    def should_save(self) -> bool:
        # Increment the phase by simulation period; subtract one save period
        # whenever we cross it.
        self._phase += 1.0 / self.simulation_hz
        save_period = 1.0 / self.save_hz
        if self._phase + 1e-9 >= save_period:
            self._phase -= save_period
            return True
        return False


@dataclass
class FrameCounter:
    saved: int = 0
    dropped_sensor_frames: int = 0
    tick_count: int = 0

    def log_progress(self, every_n_ticks: int = 20, duration_s: float = 0.0, current_t: float = 0.0) -> None:
        if every_n_ticks <= 0 or self.tick_count % every_n_ticks != 0:
            return
        if duration_s > 0:
            pct = 100.0 * current_t / duration_s
            LOGGER.info(
                "Progress: t=%.2fs / %.2fs (%.1f%%), ticks=%d, saved=%d, dropped=%d",
                current_t, duration_s, pct, self.tick_count, self.saved, self.dropped_sensor_frames,
            )
        else:
            LOGGER.info(
                "Progress: ticks=%d, saved=%d, dropped=%d",
                self.tick_count, self.saved, self.dropped_sensor_frames,
            )
