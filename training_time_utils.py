from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def start_training_timer() -> float:
    return time.perf_counter()


def write_training_time(
    save_dir: str | Path,
    *,
    method: str,
    env_name: str,
    cfg: Any,
    start_time: float,
    best_return: float | None = None,
    completed_epochs: int | None = None,
) -> dict[str, Any]:
    elapsed_sec = time.perf_counter() - start_time
    cfg_dict = getattr(cfg, "__dict__", {})
    if completed_epochs is None:
        completed_epochs = int(cfg_dict.get("epochs", 0) or 0)
    meta = {
        "method": method,
        "env_name": env_name,
        "training_time_sec": elapsed_sec,
        "training_time_min": elapsed_sec / 60.0,
        "training_time_hr": elapsed_sec / 3600.0,
        "completed_epochs": int(completed_epochs),
        "best_return": None if best_return is None else float(best_return),
        "N": int(cfg_dict.get("N", 0) or 0),
        "K": int(cfg_dict.get("K", 0) or 0),
        "T": int(cfg_dict.get("T", 0) or 0),
        "seed": int(cfg_dict.get("seed", 0) or 0),
        "device": str(cfg_dict.get("device", "")),
        "python": sys.executable,
        "platform": platform.platform(),
    }
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "training_time.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[time] training_time={meta['training_time_min']:.2f} min "
        f"({meta['training_time_hr']:.3f} hr) -> {path}"
    )
    return meta
