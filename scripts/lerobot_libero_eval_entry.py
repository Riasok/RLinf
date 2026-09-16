"""Launch LeRobot evaluation against standard or Plus LIBERO assets."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _configure_libero() -> None:
    libero_type = os.environ.get("LEROBOT_LIBERO_TYPE", "standard").lower()
    if libero_type == "plus":
        package = importlib.import_module("liberoplus")
        core = importlib.import_module("liberoplus.liberoplus")
        benchmark = importlib.import_module("liberoplus.liberoplus.benchmark")
        envs = importlib.import_module("liberoplus.liberoplus.envs")
        sys.modules["libero"] = package
        sys.modules["libero.libero"] = core
        sys.modules["libero.libero.benchmark"] = benchmark
        sys.modules["libero.libero.envs"] = envs
    elif libero_type == "standard":
        core = importlib.import_module("libero.libero")
    else:
        raise ValueError(f"Unsupported LEROBOT_LIBERO_TYPE={libero_type!r}")

    config_file = Path(os.environ["LEROBOT_LIBERO_CONFIG_FILE"]).resolve()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    core.config_file = str(config_file)
    installed_root = str(Path(core.__file__).resolve().parent)
    if not config_file.exists():
        core.set_libero_default_path(installed_root)
    try:
        configured_root = core.get_libero_path("benchmark_root")
    except Exception:
        configured_root = None
    if configured_root != installed_root:
        core.set_libero_default_path(installed_root)


if __name__ == "__main__":
    _configure_libero()
    from lerobot.scripts.lerobot_eval import main

    main()
