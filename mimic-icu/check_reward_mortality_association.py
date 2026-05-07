"""
Check whether the MIMIC vital-based health proxy is associated with 90-day mortality.

This does not validate the simulator reward as a causal clinical endpoint. It is
only a sanity check that the observed-vital proxy used to construct the latent
health factors has the expected direction: higher health proxy should be
associated with lower mortality.

Usage:
  python check_reward_mortality_association.py
  python check_reward_mortality_association.py --csv ../MIMIC/mimic_rmab_episodes_000000000000.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VITAL_COLS = ["heart_rate", "sbp", "mbp", "spo2", "resp_rate"]
VITAL_MEAN = np.array([80.0, 120.0, 80.0, 94.0, 18.0], dtype=np.float64)
VITAL_SCALE = np.array([20.0, 25.0, 15.0, 4.0, 6.0], dtype=np.float64)
VITAL_SIGN = np.array([1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
HEMO_IDX = [0, 1, 2]
RESP_IDX = [3, 4]


def _safe_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _health_proxy(row: dict[str, str]) -> float | None:
    vals = [_safe_float(row.get(c, "")) for c in VITAL_COLS]
    if any(v is None for v in vals):
        return None
    raw = np.asarray(vals, dtype=np.float64)
    normed = VITAL_SIGN * (raw - VITAL_MEAN) / VITAL_SCALE
    x_hemo = float(normed[HEMO_IDX].mean())
    x_resp = float(normed[RESP_IDX].mean())
    return max(0.5 * x_hemo + 0.5 * x_resp, 0.0)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC for larger scores predicting label=1, using average ranks."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _bootstrap_ci_diff(
    survivor: np.ndarray,
    deceased: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = rng.choice(survivor, size=len(survivor), replace=True)
        d = rng.choice(deceased, size=len(deceased), replace=True)
        diffs[b] = s.mean() - d.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def load_stay_features(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stays: dict[str, dict[str, object]] = defaultdict(lambda: {"health": [], "died": None})
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["icustay_id"]
            h = _health_proxy(row)
            if h is not None:
                stays[sid]["health"].append(h)
            died = row.get("died_90day", "")
            if died != "":
                stays[sid]["died"] = int(float(died))

    features = []
    labels = []
    for stay in stays.values():
        health = np.asarray(stay["health"], dtype=np.float64)
        died = stay["died"]
        if died is None or health.size == 0:
            continue
        features.append(float(health.mean()))
        labels.append(int(died))
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../MIMIC/mimic_rmab_episodes_000000000000.csv")
    parser.add_argument("--out", default="reward_mortality_association.png")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    health, died = load_stay_features(csv_path)
    survivor = health[died == 0]
    deceased = health[died == 1]

    diff = float(survivor.mean() - deceased.mean())
    ci_lo, ci_hi = _bootstrap_ci_diff(survivor, deceased, args.bootstrap, args.seed)
    corr = float(np.corrcoef(health, died)[0, 1])
    auc_mortality = _auc(-health, died)

    print(f"CSV: {csv_path}")
    print(f"ICU stays used: {len(health)}")
    print(f"90-day mortality rate: {died.mean():.3f}")
    print(f"Mean health proxy, survivors:    {survivor.mean():.4f} +/- {survivor.std():.4f}")
    print(f"Mean health proxy, non-survivors:{deceased.mean():.4f} +/- {deceased.std():.4f}")
    print(f"Difference survivor - non-survivor: {diff:.4f}")
    print(f"Bootstrap 95% CI for difference: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Pearson corr(health_proxy, died_90day): {corr:.4f}")
    print(f"AUC using -health_proxy to predict died_90day: {auc_mortality:.4f}")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.boxplot(
        [survivor, deceased],
        labels=["Survived", "Died"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8f5", "edgecolor": "black"},
        medianprops={"color": "black"},
    )
    ax.set_ylabel("Mean vital health proxy")
    ax.set_title("MIMIC health proxy vs. 90-day mortality")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
