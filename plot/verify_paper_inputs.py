"""Validate saved result files used to regenerate the paper figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MAIN_SPECS = [
    ("synthetic-stationary", 20, 5),
    ("synthetic-drifting", 20, 5),
    ("mimic-icu", 20, 5),
    ("synthetic-stationary", 40, 8),
    ("synthetic-drifting", 40, 8),
    ("mimic-icu", 40, 8),
    ("synthetic-stationary", 100, 20),
    ("synthetic-drifting", 100, 20),
    ("mimic-icu", 100, 20),
]
ACTOR_SPECS = [
    "synthetic-stationary/actor_ablation_seed0_N20_K5..npz",
    "synthetic-drifting/actor_ablation_seed0_N20_K5..npz",
    "mimic-icu/actor_ablation_seed0_N20_K5.npz",
]
EXPECTED_METHODS = ["random", "greedy", "neurwin", "ppo", "dpmd", "oracle_lookahead"]


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> int:
    return int(np.asarray(data[key]).item())


def validate_main() -> None:
    for env_dir, n_arms, budget in MAIN_SPECS:
        path = ROOT / env_dir / f"comparison_N{n_arms}_K{budget}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        methods = [str(x) for x in data["method_names"]]
        if methods != EXPECTED_METHODS:
            raise ValueError(f"{path}: unexpected method order {methods}")
        checks = {
            "N": n_arms,
            "K": budget,
            "T": 100,
            "seed": 42,
            "n_episodes": 100,
        }
        for key, expected in checks.items():
            if key not in data:
                raise ValueError(f"{path}: missing metadata key {key}")
            actual = _scalar(data, key)
            if actual != expected:
                raise ValueError(f"{path}: {key}={actual}, expected {expected}")
        for method in EXPECTED_METHODS:
            values = np.asarray(data[method], dtype=float)
            if values.shape != (100,):
                raise ValueError(f"{path}: {method} shape {values.shape}, expected (100,)")
    print("[ok] main comparison inputs")


def validate_actor() -> None:
    for rel in ACTOR_SPECS:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        names = [str(x) for x in data["method_names"]] if "method_names" in data else list(data.files)
        normalized = {"dpmd_joint_N" if name == "joint_N" else name for name in names}
        required = {"mlp_actor", "dpmd_joint_N", "dpmd"}
        if not required.issubset(normalized):
            raise ValueError(f"{path}: missing actor ablation methods; found {names}")
    print("[ok] actor ablation inputs")


def main() -> None:
    validate_main()
    validate_actor()
    print("[ok] saved paper inputs are complete")


if __name__ == "__main__":
    main()
