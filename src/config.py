"""
config.py
---------
Loads the YAML configuration file and exposes it as a hierarchy of
dataclass-like dictionaries. We intentionally keep this very light - the YAML
is the source of truth, and the rest of the code just dereferences keys.

The only place we *do* coerce values is in :func:`load_config` where we
validate a couple of required keys and turn relative output paths into
absolute paths.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL_KEYS = (
    "simulation",
    "ugv",
    "uav",
    "relay_cameras",
    "save",
)


def _validate(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"Config is missing required top-level keys: {missing}"
        )

    sim = cfg["simulation"]
    if sim.get("synchronous_mode") and not sim.get("fixed_delta_seconds"):
        raise ValueError(
            "synchronous_mode=True requires a non-zero fixed_delta_seconds"
        )

    ugv = cfg["ugv"]
    if ugv.get("sensor_mode") not in {"rgbd", "stereo"}:
        raise ValueError(
            f"ugv.sensor_mode must be 'rgbd' or 'stereo', got {ugv.get('sensor_mode')!r}"
        )

    uav = cfg["uav"]
    if uav.get("mode") not in {"follow_ugv", "hover", "waypoint"}:
        raise ValueError(
            f"uav.mode must be follow_ugv / hover / waypoint, got {uav.get('mode')!r}"
        )


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load and validate a YAML config file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}")

    cfg = deepcopy(raw)
    _validate(cfg)

    # Resolve output_dir to absolute, but anchor relative paths to the *config
    # file's parent's parent* (i.e. the project root) so it is independent of
    # the user's current working directory.
    sim = cfg["simulation"]
    out = Path(sim["output_dir"])
    if not out.is_absolute():
        project_root = path.parent.parent  # configs/default.yaml -> project root
        out = (project_root / out).resolve()
    sim["output_dir"] = str(out)
    cfg["_config_path"] = str(path)
    return cfg


def dump_config(cfg: Dict[str, Any], dest: str | os.PathLike) -> None:
    """Persist a config snapshot (sans the private _config_path key)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    snap = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with dest.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(snap, fp, sort_keys=False, allow_unicode=True)
