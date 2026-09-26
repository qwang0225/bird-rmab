"""
cluster_sweep.py

Step 1: Fit AR(1) per ICU stay from MIMIC vitals.
Step 2: Sweep K-means for M=2..6 clusters, report elbow + silhouette.
Step 3: Print per-cluster clinical profiles for each M.

Usage:
    python cluster_sweep.py
    python cluster_sweep.py --max_stays 3000   # fast test on subset

Output:
    cluster_sweep.png   -- elbow + silhouette plots
    Console prints per-cluster profiles for each M value.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Vitals config ──────────────────────────────────────────────────────────────
VITAL_COLS  = ["heart_rate", "sbp", "mbp", "spo2", "resp_rate"]
D           = len(VITAL_COLS)
VITAL_MEAN  = np.array([80.0, 120.0, 80.0, 94.0, 18.0])
VITAL_SCALE = np.array([20.0,  25.0, 15.0,  4.0,  6.0])
VITAL_SIGN  = np.array([ 1.0,   1.0,  1.0,  1.0, -1.0])  # high RR = bad
MIN_HOURS   = 12   # minimum complete vital rows to include a stay


# ── Data loading ───────────────────────────────────────────────────────────────
def load_stays(csv_path: str, max_stays: int = 0) -> dict:
    stays: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["icustay_id"]
            if max_stays and len(stays) >= max_stays and sid not in stays:
                continue
            if sid not in stays:
                stays[sid] = {"x": [], "a": [], "died": int(row.get("died_90day", 0))}
            try:
                vals = np.array([float(row[c]) for c in VITAL_COLS])
                normed = VITAL_SIGN * (vals - VITAL_MEAN) / VITAL_SCALE
                stays[sid]["x"].append(normed.astype(np.float32))
            except (ValueError, KeyError):
                stays[sid]["x"].append(None)
            stays[sid]["a"].append(int(row.get("action", 0)))
    return stays


# ── AR(1) fitting ──────────────────────────────────────────────────────────────
def fit_stay(stay: dict) -> dict | None:
    """
    OLS per vital dim: x_{t+1,d} = alpha_d * x_{t,d} + beta_d * a_t + drift_d + eps
    Returns scalar summaries: mean_alpha, mean_beta, mean_drift, mean_sigma_w
    """
    xs, actions = stay["x"], stay["a"]
    pairs = [(xs[t], xs[t+1], actions[t]) for t in range(len(xs)-1)
             if xs[t] is not None and xs[t+1] is not None]
    if len(pairs) < MIN_HOURS:
        return None

    X_cur  = np.stack([p[0] for p in pairs])
    X_next = np.stack([p[1] for p in pairs])
    A      = np.array([p[2] for p in pairs], dtype=np.float64)

    alphas, betas, drifts, sigmas = [], [], [], []
    for d in range(D):
        Z  = np.column_stack([X_cur[:, d], A, np.ones(len(A))])
        Yd = X_next[:, d]
        try:
            coef, _, _, _ = np.linalg.lstsq(Z, Yd, rcond=None)
        except np.linalg.LinAlgError:
            return None
        alpha_d, beta_d, drift_d = coef
        resid = Yd - Z @ coef
        alphas.append(float(np.clip(alpha_d, 0.3, 0.999)))
        betas.append(float(beta_d))
        drifts.append(float(drift_d))
        sigmas.append(float(np.std(resid) + 1e-4))

    return {
        "alpha":   float(np.mean(alphas)),   # mean persistence across vitals
        "beta":    float(np.mean(betas)),    # mean treatment responsiveness
        "drift":   float(np.mean(drifts)),   # mean baseline health trend
        "sigma_w": float(np.mean(sigmas)),   # mean residual noise
        "n_acts":  int(sum(actions)),
        "died":    stay["died"],
        "length":  len(xs),
    }


# ── K-means ────────────────────────────────────────────────────────────────────
def kmeans(F: np.ndarray, M: int, seed: int = 42, n_init: int = 10) -> tuple[np.ndarray, float]:
    """Returns (labels, inertia). Multi-restart k-means."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = F[rng.choice(len(F), M, replace=False)].copy()
        for _ in range(200):
            dists   = np.stack([np.sum((F - c)**2, axis=1) for c in centers], axis=1)
            labels  = np.argmin(dists, axis=1)
            new_c   = np.array([F[labels == m].mean(0) if (labels == m).any()
                                else centers[m] for m in range(M)])
            if np.allclose(centers, new_c, atol=1e-8):
                break
            centers = new_c
        inertia = sum(np.sum((F[labels == m] - centers[m])**2)
                      for m in range(M) if (labels == m).any())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels, best_inertia


def silhouette(F: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score (simplified, O(n^2) — fast enough for <20k stays)."""
    M = labels.max() + 1
    scores = []
    # subsample if large
    if len(F) > 5000:
        idx = np.random.default_rng(0).choice(len(F), 5000, replace=False)
        F, labels = F[idx], labels[idx]
    for i in range(len(F)):
        m_i = labels[i]
        same  = F[labels == m_i]
        a = np.mean(np.sqrt(np.sum((same - F[i])**2, axis=1) + 1e-12)) if len(same) > 1 else 0.0
        bs = []
        for m in range(M):
            if m == m_i:
                continue
            other = F[labels == m]
            if len(other):
                bs.append(np.mean(np.sqrt(np.sum((other - F[i])**2, axis=1) + 1e-12)))
        b = min(bs) if bs else 0.0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)
    return float(np.mean(scores))


# ── Cluster profiling ──────────────────────────────────────────────────────────
def print_cluster_profile(records: list[dict], labels: np.ndarray, M: int):
    print(f"\n{'='*60}")
    print(f"  M = {M} clusters")
    print(f"{'='*60}")
    # Sort clusters by mean beta ascending (non-responder → high-responder)
    cluster_beta = {m: np.mean([r["beta"] for r, l in zip(records, labels) if l == m])
                    for m in range(M)}
    order = sorted(range(M), key=lambda m: cluster_beta[m])

    for rank, m in enumerate(order):
        idx = [i for i, l in enumerate(labels) if l == m]
        group = [records[i] for i in idx]
        alphas  = [r["alpha"]   for r in group]
        betas   = [r["beta"]    for r in group]
        drifts  = [r["drift"]   for r in group]
        sigmas  = [r["sigma_w"] for r in group]
        mort    = np.mean([r["died"]   for r in group])
        acts    = np.mean([r["n_acts"] for r in group])
        length  = np.mean([r["length"] for r in group])
        print(f"  Cluster {rank} (raw={m}, n={len(group)}):")
        print(f"    alpha (persistence) : {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")
        print(f"    beta  (tx response) : {np.mean(betas):.3f} ± {np.std(betas):.3f}")
        print(f"    drift (health trend): {np.mean(drifts):.3f} ± {np.std(drifts):.3f}")
        print(f"    sigma_w (noise)     : {np.mean(sigmas):.3f} ± {np.std(sigmas):.3f}")
        print(f"    mortality rate      : {mort*100:.1f}%")
        print(f"    avg interventions   : {acts:.1f}")
        print(f"    avg stay length (h) : {length:.0f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=
        "../MIMIC/mimic_rmab_episodes_000000000000.csv")
    parser.add_argument("--max_stays", type=int, default=0,
                        help="0 = all (slow). Use e.g. 5000 for a quick run.")
    parser.add_argument("--M_max",    type=int, default=6)
    parser.add_argument("--out",      default="cluster_sweep.png")
    args = parser.parse_args()

    print(f"Loading stays from {args.csv} ...")
    stays = load_stays(args.csv, max_stays=args.max_stays)
    print(f"  {len(stays)} stays loaded")

    print(f"Fitting AR(1) per stay (need >= {MIN_HOURS} complete vital rows) ...")
    records = []
    for sid, stay in stays.items():
        r = fit_stay(stay)
        if r is not None:
            records.append(r)
    print(f"  {len(records)} stays fit successfully")

    # Feature matrix for clustering: [alpha, beta, drift, sigma_w]
    F_raw = np.array([[r["alpha"], r["beta"], r["drift"], r["sigma_w"]]
                      for r in records], dtype=np.float64)
    F = (F_raw - F_raw.mean(0)) / (F_raw.std(0) + 1e-6)

    M_vals      = list(range(2, args.M_max + 1))
    inertias    = []
    silhouettes = []
    all_labels  = {}

    for M in M_vals:
        print(f"  K-means M={M} ...", end=" ", flush=True)
        labels, inertia = kmeans(F, M)
        sil = silhouette(F, labels)
        print(f"inertia={inertia:.1f}  silhouette={sil:.3f}")
        inertias.append(inertia)
        silhouettes.append(sil)
        all_labels[M] = labels

    # ── Plot elbow + silhouette ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(M_vals, inertias, "o-", color="steelblue")
    axes[0].set_xlabel("M (number of clusters)")
    axes[0].set_ylabel("K-means inertia (lower = tighter)")
    axes[0].set_title("Elbow plot")
    axes[0].set_xticks(M_vals)

    axes[1].plot(M_vals, silhouettes, "o-", color="coral")
    axes[1].set_xlabel("M (number of clusters)")
    axes[1].set_ylabel("Mean silhouette score (higher = better)")
    axes[1].set_title("Silhouette score")
    axes[1].set_xticks(M_vals)

    best_M = M_vals[int(np.argmax(silhouettes))]
    axes[1].axvline(best_M, color="gray", linestyle="--", label=f"best M={best_M}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nPlot saved to {args.out}")

    # ── Print cluster profiles for each M ─────────────────────────────────────
    for M in M_vals:
        print_cluster_profile(records, all_labels[M], M)

    print(f"\n>>> Silhouette scores: { {M: round(s,3) for M,s in zip(M_vals,silhouettes)} }")
    print(f">>> Best M by silhouette: M={best_M}")


if __name__ == "__main__":
    main()
