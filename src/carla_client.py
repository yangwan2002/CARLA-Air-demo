"""
carla_client.py
---------------
Thin wrapper around the CARLA python client that:

* connects with timeout + retries,
* (optionally) loads the requested map and applies weather,
* switches the world into synchronous mode with a fixed delta,
* tracks the previously-applied world settings so we can restore them on exit.

Used as a context manager:

    with CarlaWorld(cfg["simulation"]) as cw:
        cw.world.tick()
        ...
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

try:
    import carla  # type: ignore
except ImportError as e:  # pragma: no cover
    carla = None  # type: ignore
    _import_error: Optional[BaseException] = e
else:
    _import_error = None


LOGGER = logging.getLogger(__name__)


def _require_carla() -> None:
    if carla is None:
        raise ImportError(
            "The `carla` python package is not importable. Make sure the CARLA "
            "PythonAPI egg/wheel matching your CARLA-Air install is on PYTHONPATH."
        ) from _import_error


class CarlaWorld:
    """Manage the CARLA client + world lifecycle."""

    def __init__(self, sim_cfg: Dict[str, Any], connect_timeout_s: float = 10.0) -> None:
        _require_carla()
        self.sim_cfg = sim_cfg
        self.connect_timeout_s = connect_timeout_s

        self.client: Optional["carla.Client"] = None
        self.world: Optional["carla.World"] = None
        self.tm: Optional["carla.TrafficManager"] = None
        self._original_settings: Optional["carla.WorldSettings"] = None
        self._original_tm_sync: Optional[bool] = None

    # ------------------------------------------------------------------
    # context-manager API
    # ------------------------------------------------------------------
    def __enter__(self) -> "CarlaWorld":
        self.connect()
        self.apply_map_and_weather()
        self.enable_synchronous_mode()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    def connect(self) -> None:
        host = self.sim_cfg.get("carla_host", "localhost")
        port = int(self.sim_cfg.get("carla_port", 2000))

        LOGGER.info("Connecting to CARLA at %s:%d (timeout=%.1fs)", host, port, self.connect_timeout_s)
        client = carla.Client(host, port)
        client.set_timeout(self.connect_timeout_s)

        # Sanity-check the connection by reading the server version.
        try:
            server_version = client.get_server_version()
            client_version = client.get_client_version()
        except RuntimeError as e:
            raise ConnectionError(
                f"Could not reach CARLA at {host}:{port}. Is the CARLA-Air server running?"
            ) from e

        LOGGER.info("Connected. CARLA server=%s, client=%s", server_version, client_version)
        if server_version != client_version:
            LOGGER.warning(
                "CARLA server/client version mismatch (server=%s, client=%s). "
                "Some APIs may behave differently.",
                server_version, client_version,
            )
        self.client = client

    def apply_map_and_weather(self) -> None:
        assert self.client is not None
        target_map = self.sim_cfg.get("map")
        world = self.client.get_world()
        if target_map:
            current_map_name = world.get_map().name.split("/")[-1]
            if current_map_name != target_map:
                LOGGER.info("Loading map %s (current=%s)", target_map, current_map_name)
                world = self.client.load_world(target_map)
                # load_world() returns immediately, give the simulator a moment
                # to fully transition before we touch its settings.
                time.sleep(2.0)
            else:
                LOGGER.info("Map already loaded: %s", target_map)
        self.world = world

        weather_cfg = self.sim_cfg.get("weather")
        if weather_cfg:
            LOGGER.info("Applying weather override: %s", weather_cfg)
            try:
                weather = world.get_weather()
                for k, v in weather_cfg.items():
                    if hasattr(weather, k):
                        setattr(weather, k, float(v))
                    else:
                        LOGGER.warning("Weather attribute %r not recognized; skipping.", k)
                world.set_weather(weather)
            except Exception as e:  # pragma: no cover - defensive
                LOGGER.warning("Failed to apply weather override: %s", e)

    def enable_synchronous_mode(self) -> None:
        assert self.client is not None and self.world is not None
        if not self.sim_cfg.get("synchronous_mode", True):
            LOGGER.info("synchronous_mode disabled in config; not modifying CARLA world settings.")
            return

        delta = float(self.sim_cfg.get("fixed_delta_seconds", 0.05))
        self._original_settings = self.world.get_settings()

        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = delta
        # Substepping helps physics stay stable at higher delta_seconds.
        new_settings.substepping = True
        new_settings.max_substep_delta_time = 0.01
        new_settings.max_substeps = 10
        self.world.apply_settings(new_settings)

        # Traffic manager must also be put into sync mode, otherwise autopilot
        # vehicles will desync.
        try:
            self.tm = self.client.get_trafficmanager()
            self._original_tm_sync = self.tm.synchronous_mode  # type: ignore[attr-defined]
            self.tm.set_synchronous_mode(True)
            seed = self.sim_cfg.get("seed")
            if seed is not None:
                self.tm.set_random_device_seed(int(seed))
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Could not configure TrafficManager sync mode: %s", e)

        LOGGER.info(
            "CARLA synchronous mode ON, fixed_delta_seconds=%.4f, substepping enabled",
            delta,
        )

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Restore CARLA to its previous mode. Safe to call multiple times."""
        if self.world is not None and self._original_settings is not None:
            try:
                LOGGER.info("Restoring original CARLA world settings (async mode).")
                self.world.apply_settings(self._original_settings)
            except Exception as e:
                LOGGER.warning("Failed to restore CARLA world settings: %s", e)
            self._original_settings = None

        if self.tm is not None and self._original_tm_sync is not None:
            try:
                self.tm.set_synchronous_mode(bool(self._original_tm_sync))
            except Exception as e:
                LOGGER.warning("Failed to restore TrafficManager mode: %s", e)
            self._original_tm_sync = None

        self.tm = None
        self.world = None
        self.client = None
