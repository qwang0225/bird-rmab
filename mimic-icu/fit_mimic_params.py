"""
fit_mimic_params.py

Reads mimic_rmab_episodes_000000000000.csv, fits per-stay AR(1) dynamics for
5 normalized vitals, clusters stays into M=3 patient types, and saves
mimic_params.json for use by env.py.

Usage:
    python fit_mimic_params.py [--csv PATH] [--out mimic_params.json] [--M 4]

Vitals used (5D state):
    0: heart_rate    normalized as (hr  - 80) / 20
    1: sbp           normalized as (sbp - 120) / 25
    2: mbp           normalized as (mbp - 80)  / 15
    3: spo2          normalized as (spo2 - 94) / 4    <-- most critical
    4: resp_rate     normalized as -(rr - 18)  / 6    (high RR = bad)

Temp and glucose excluded (67%/68% missing) — they become the POMDP's
"hidden signal": the agent never observes them, making true health state
partially observable even with perfect vital observations.

AR(1) per stay (scalar alpha, 5D beta):
    x_{t+1} = alpha * x_t + beta * a_t + drift + eps

Fit via OLS on stays with >= MIN_HOURS observations and at least 1 action.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np

VITAL_COLS = ["heart_rate", "sbp", "mbp", "spo2", "resp_rate"]
D = len(VITAL_COLS)
MIN_HOURS = 12   # minimum hours to use a stay for fitting

# Normalization: healthy  → 0,  sick → negative
VITAL_MEAN  = np.array([80.0, 120.0, 80.0, 94.0, 18.0], dtype=np.float64)
VITAL_SCALE = np.array([20.0,  25.0, 15.0,  4.0,  6.0], dtype=np.float64)
VITAL_SIGN  = np.array([1.0,    1.0,  1.0,  1.0, -1.0], dtype=np.float64)  # resp_rate: high=bad


def normalize(raw: np.ndarray) -> np.ndarray:
    """raw shape (T, D) → normalized (T, D)."""
    return VITAL_SIGN * (raw - VITAL_MEAN) / VITAL_SCALE


def parse_row(row: dict) -> tuple[np.ndarray | None, int]:
    """Return (normalized_vitals [D], action) or (None, action) if any vital missing."""
    try:
        vals = np.array([float(row[c]) for c in VITAL_COLS], dtype=np.float64)
    except (ValueError, KeyError):
        return None, int(row.get("action", 0))
    normed = VITAL_SIGN * (vals - VITAL_MEAN) / VITAL_SCALE
    return normed.astype(np.float32), int(row.get("action", 0))


def load_stays(csv_path: str) -> dict[str, dict]:
    """Load all stays. Returns {stay_id: {'x': list[np.ndarray|None], 'a': list[int]}}."""
    stays: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["icustay_id"]
            if sid not in stays:
                stays[sid] = {"x": [], "a": [], "died": int(row.get("died_90day", 0))}
            x, a = parse_row(row)
            stays[sid]["x"].append(x)
            stays[sid]["a"].append(a)
    return stays


def fit_stay_ar1(xs: list, actions: list) -> dict | None:
    """
    OLS fit: x_{t+1} = alpha * x_t + beta * a_t + drift + eps
    Returns dict with alpha, beta (D,), drift (D,), sigma_w (D,) or None if too short.
    We fit a common scalar alpha (mean of per-dim estimates) for simplicity.
    """
    pairs = [(xs[t], xs[t+1], actions[t]) for t in range(len(xs)-1)
             if xs[t] is not None and xs[t+1] is not None]
    if len(pairs) < MIN_HOURS:
        return None

    X_cur  = np.stack([p[0] for p in pairs])     # (T, D)
    X_next = np.stack([p[1] for p in pairs])     # (T, D)
    A      = np.array([p[2] for p in pairs], dtype=np.float64)  # (T,)

    # Design matrix: [x_t, a_t, 1] for each dim separately
    # x_{t+1,d} = alpha_d * x_{t,d} + beta_d * a_t + drift_d
    results = {"alpha": [], "beta": [], "drift": [], "sigma_w": []}
    for d in range(D):
        Xd = X_cur[:, d]
        Yd = X_next[:, d]
        # [x_t, a_t, 1]
        Z = np.column_stack([Xd, A, np.ones(len(Xd))])
        try:
            coef, _, _, _ = np.linalg.lstsq(Z, Yd, rcond=None)
        except np.linalg.LinAlgError:
            return None
        alpha_d, beta_d, drift_d = coef
        resid = Yd - Z @ coef
        results["alpha"].append(float(np.clip(alpha_d, 0.5, 0.99)))
        results["beta"].append(float(beta_d))
        results["drift"].append(float(drift_d))
        results["sigma_w"].append(float(np.std(resid) + 1e-4))

    return results


def cluster_stays(stay_params: list[dict], M: int = 4, seed: int = 42) -> np.ndarray:
    """K-means cluster stays by (alpha_mean, beta_mean, drift_mean). Returns labels."""
    features = []
    for p in stay_params:
        alpha_m = float(np.mean(p["alpha"]))
        beta_m  = float(np.mean(p["beta"]))
        drift_m = float(np.mean(p["drift"]))
        sigma_m = float(np.mean(p["sigma_w"]))
        features.append([alpha_m, beta_m, drift_m, sigma_m])
    F = np.array(features, dtype=np.float64)

    # Normalize features for clustering
    F_norm = (F - F.mean(0)) / (F.std(0) + 1e-6)

    rng = np.random.default_rng(seed)
    # Simple k-means
    centers = F_norm[rng.choice(len(F_norm), M, replace=False)]
    for _ in range(100):
        dists = np.stack([np.sum((F_norm - c)**2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([F_norm[labels == m].mean(0) if (labels == m).any()
                                else centers[m] for m in range(M)])
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return labels


def compute_type_params(stay_params: list[dict], labels: np.ndarray,
                        M: int, mortalities: list[int]) -> dict:
    """
    For each cluster, compute type means and stds.
    Types are sorted by mean beta (treatment responsiveness), ascending.
    Type 0 = least responsive (worst for intervention), Type 3 = most responsive.
    """
    type_data: dict[int, list] = defaultdict(list)
    for i, p in enumerate(stay_params):
        type_data[int(labels[i])].append(p)

    type_params = {}
    type_stats = {}
    for m in range(M):
        group = type_data[m]
        if not group:
            continue
        alpha_bars = np.array([np.mean(p["alpha"]) for p in group])
        beta_bars  = np.array([np.mean(p["beta"])  for p in group])
        drift_bars = np.array([np.mean(p["drift"]) for p in group])
        sigma_ws   = np.array([np.mean(p["sigma_w"]) for p in group])

        type_stats[m] = {
            "count":    len(group),
            "alpha_bar": float(np.median(alpha_bars)),
            "beta_bar":  float(np.median(beta_bars)),
            "drift_bar": float(np.median(drift_bars)),
            "sigma_w":   float(np.median(sigma_ws)),
            "alpha_std": float(np.std(alpha_bars)),
            "beta_std":  float(np.std(beta_bars)),
        }

    # Sort by beta_bar ascending → type 0 = non-responder, type M-1 = best responder
    sorted_types = sorted(type_stats.keys(), key=lambda m: type_stats[m]["beta_bar"])
    remap = {old: new for new, old in enumerate(sorted_types)}

    final = {"M": M, "D": D, "vital_names": VITAL_COLS,
             "vital_mean": VITAL_MEAN.tolist(), "vital_scale": VITAL_SCALE.tolist(),
             "vital_sign": VITAL_SIGN.tolist(), "types": {}}

    type_names = ["non-responder", "slow-responder", "responder"]
    for old_m, new_m in remap.items():
        s = type_stats[old_m]
        final["types"][str(new_m)] = {
            "name":      type_names[new_m],
            "count":     s["count"],
            "alpha_bar": round(s["alpha_bar"], 4),
            "beta_bar":  round(s["beta_bar"],  4),
            "drift_bar": round(s["drift_bar"], 4),
            "sigma_w":   round(s["sigma_w"],   4),
            "alpha_std": round(s["alpha_std"], 4),
            "beta_std":  round(s["beta_std"],  4),
        }

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=
        "../MIMIC/mimic_rmab_episodes_000000000000.csv")
    parser.add_argument("--out", default="mimic_params.json")
    parser.add_argument("--M",   type=int, default=3)
    parser.add_argument("--max_stays", type=int, default=0,
                        help="0 = all stays (slow); set e.g. 5000 for fast test")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    stays = load_stays(args.csv)
    print(f"  {len(stays)} unique ICU stays loaded")

    stay_ids = sorted(stays.keys())
    if args.max_stays > 0:
        stay_ids = stay_ids[:args.max_stays]
        print(f"  Using first {len(stay_ids)} stays")

    print("Fitting AR(1) per stay ...")
    fitted_ids = []
    fitted_params = []
    mortalities = []
    for i, sid in enumerate(stay_ids):
        if i % 2000 == 0:
            print(f"  {i}/{len(stay_ids)} ...", flush=True)
        result = fit_stay_ar1(stays[sid]["x"], stays[sid]["a"])
        if result is not None:
            fitted_ids.append(sid)
            fitted_params.append(result)
            mortalities.append(stays[sid]["died"])

    print(f"  {len(fitted_params)} stays with >= {MIN_HOURS}h complete vitals")

    print(f"Clustering into M={args.M} types ...")
    labels = cluster_stays(fitted_params, M=args.M)
    for m in range(args.M):
        print(f"  Type {m}: {(labels == m).sum()} stays")

    print("Computing type parameters ...")
    params = compute_type_params(fitted_params, labels, args.M, mortalities)

    print("\n=== Fitted Type Parameters ===")
    for m in range(args.M):
        t = params["types"][str(m)]
        print(f"  Type {m} ({t['name']}, n={t['count']}): "
              f"alpha_bar={t['alpha_bar']:.3f}  beta_bar={t['beta_bar']:.3f}  "
              f"drift_bar={t['drift_bar']:.3f}  sigma_w={t['sigma_w']:.3f}")

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
