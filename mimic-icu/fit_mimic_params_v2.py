"""
fit_mimic_params_v2.py

Fits parameters for the redesigned 3D-latent MIMICv2 environment from
mimic_rmab_episodes_*.csv.

New vs v1:
  - 2D AR(1) per stay: separate hemo and resp proxy states with cross-coupling
  - Loading matrix C (5x2): how each vital loads on (x_hemo, x_resp)
  - Sparse metabolic obs: temperature and glucose loading onto x_meta (proxy)
  - M=4 patient types clustered by (beta_hemo, beta_resp, drift_hemo, drift_resp)

Output: mimic_params_v2.json

Usage:
    python fit_mimic_params_v2.py
    python fit_mimic_params_v2.py --csv path/to/csv --M 4 --out mimic_params_v2.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

# ── Vital columns (always present) ──────────────────────────────────────────
VITAL_COLS  = ["heart_rate", "sbp", "mbp", "spo2", "resp_rate"]
META_COLS   = ["temperature", "glucose"]   # sparse (~33% / ~32% present)
D_VITAL     = len(VITAL_COLS)
D_META      = len(META_COLS)
MIN_HOURS   = 12

# Normalization: healthy → positive, sick → negative / near zero
VITAL_MEAN  = np.array([80.0, 120.0,  80.0, 94.0, 18.0], dtype=np.float64)
VITAL_SCALE = np.array([20.0,  25.0,  15.0,  4.0,  6.0], dtype=np.float64)
VITAL_SIGN  = np.array([ 1.0,   1.0,   1.0,  1.0, -1.0], dtype=np.float64)  # RR: high=bad

META_MEAN   = np.array([37.0, 120.0], dtype=np.float64)
META_SCALE  = np.array([ 1.0,  40.0], dtype=np.float64)
META_SIGN   = np.array([ 1.0,  -1.0], dtype=np.float64)  # glucose: high=bad

# Grouping for proxy construction
HEMO_IDX = [0, 1, 2]   # HR, SBP, MBP  → x_hemo proxy
RESP_IDX = [3, 4]       # SpO2, RR      → x_resp proxy


def normalize_vitals(raw: np.ndarray) -> np.ndarray:
    """(T, D_VITAL) raw → normalized."""
    return VITAL_SIGN * (raw - VITAL_MEAN) / VITAL_SCALE


def normalize_meta(raw: np.ndarray) -> np.ndarray:
    """(T, D_META) raw (with NaN for missing) → normalized (NaN preserved)."""
    return META_SIGN * (raw - META_MEAN) / META_SCALE


def hemo_proxy(normed: np.ndarray) -> np.ndarray:
    """(T, D_VITAL) → (T,) mean of hemo vitals."""
    return normed[:, HEMO_IDX].mean(axis=1)


def resp_proxy(normed: np.ndarray) -> np.ndarray:
    """(T, D_VITAL) → (T,) mean of resp vitals."""
    return normed[:, RESP_IDX].mean(axis=1)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_stays(csv_path: str) -> dict:
    stays: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["icustay_id"]
            if sid not in stays:
                stays[sid] = {"vitals": [], "meta": [], "actions": [], "died": 0}

            # Parse vitals
            try:
                v = np.array([float(row[c]) for c in VITAL_COLS], dtype=np.float64)
            except (ValueError, KeyError):
                v = None

            # Parse meta (NaN if missing)
            meta_row = []
            for c in META_COLS:
                try:
                    meta_row.append(float(row[c]))
                except (ValueError, KeyError):
                    meta_row.append(np.nan)

            stays[sid]["vitals"].append(v)
            stays[sid]["meta"].append(meta_row)
            stays[sid]["actions"].append(int(row.get("action", 0)))
            stays[sid]["died"] = int(row.get("died_90day", 0))
    return stays


# ── Per-stay 2D AR(1) fitting ─────────────────────────────────────────────────

def fit_stay_2d(stay: dict) -> dict | None:
    """
    Fit coupled 2D AR(1):
      x_h(t+1) = a_hh*x_h(t) + a_rh*x_r(t) + b_h*A(t) + d_h + eps_h
      x_r(t+1) = a_hr*x_h(t) + a_rr*x_r(t) + b_r*A(t) + d_r + eps_r

    Also fits loading matrix row-by-row: y_d = c_h*x_h + c_r*x_r + eps.
    Returns None if stay is too short.
    """
    vitals_list = stay["vitals"]
    meta_list   = stay["meta"]
    actions     = stay["actions"]

    # Build sequence of (normed_vitals, action) pairs where vitals not None
    pairs = []
    for t in range(len(vitals_list) - 1):
        if vitals_list[t] is None or vitals_list[t + 1] is None:
            continue
        vt  = normalize_vitals(vitals_list[t][None])[0]
        vt1 = normalize_vitals(vitals_list[t + 1][None])[0]
        a   = float(actions[t])
        pairs.append((vt, vt1, a))

    if len(pairs) < MIN_HOURS:
        return None

    V_t  = np.stack([p[0] for p in pairs])   # (T, D)
    V_t1 = np.stack([p[1] for p in pairs])
    A    = np.array([p[2] for p in pairs])    # (T,)

    x_h  = hemo_proxy(V_t)
    x_r  = resp_proxy(V_t)
    x_h1 = hemo_proxy(V_t1)
    x_r1 = resp_proxy(V_t1)

    # Design matrix for 2D AR(1): [x_h, x_r, a, 1]
    Z = np.column_stack([x_h, x_r, A, np.ones(len(x_h))])

    def ols(y):
        try:
            coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            resid = y - Z @ coef
            return coef, float(np.std(resid) + 1e-4)
        except np.linalg.LinAlgError:
            return None, None

    c_h, sw_h = ols(x_h1)
    c_r, sw_r = ols(x_r1)
    if c_h is None or c_r is None:
        return None

    result = {
        "alpha_hemo": float(np.clip(c_h[0], 0.30, 0.99)),
        "c_rh":       float(c_h[1]),                          # resp → hemo coupling
        "beta_hemo":  float(c_h[2]),
        "drift_hemo": float(c_h[3]),
        "sigma_hemo": sw_h,
        "alpha_resp": float(np.clip(c_r[1], 0.30, 0.99)),
        "c_hr":       float(c_r[0]),                          # hemo → resp coupling
        "beta_resp":  float(c_r[2]),
        "drift_resp": float(c_r[3]),
        "sigma_resp": sw_r,
    }

    # ── Loading matrix: regress each vital on (x_h, x_r) ──────────────────
    Z2 = np.column_stack([x_h, x_r, np.ones(len(x_h))])
    loadings = []
    for d in range(D_VITAL):
        try:
            coef2, _, _, _ = np.linalg.lstsq(Z2, V_t[:, d], rcond=None)
            loadings.append([float(coef2[0]), float(coef2[1])])
        except np.linalg.LinAlgError:
            loadings.append([0.0, 0.0])
    result["loading_matrix"] = loadings   # (D_VITAL, 2)

    # ── Sparse meta: regress normed temp/glucose on (x_h, x_r) ───────────
    meta_arr = np.array(meta_list, dtype=np.float64)            # (T_raw, D_META)
    meta_norm = normalize_meta(meta_arr)

    meta_loadings = []
    meta_obs_counts = []
    for d in range(D_META):
        col = meta_norm[:len(vitals_list), d]
        observed = ~np.isnan(col)
        meta_obs_counts.append(int(observed.sum()))
        if observed.sum() < 3:
            meta_loadings.append([0.0, 0.0])
            continue
        # Align with (x_h, x_r): use pairs where both vitals and meta are present
        pair_idx = [i for i, (vt_, vt1_, a_) in enumerate(pairs)
                    if i < len(col) and not np.isnan(col[i])]
        if len(pair_idx) < 3:
            meta_loadings.append([0.0, 0.0])
            continue
        y_m = col[[i for i in pair_idx]]
        Zm  = np.column_stack([x_h[pair_idx], x_r[pair_idx], np.ones(len(pair_idx))])
        try:
            coef_m, _, _, _ = np.linalg.lstsq(Zm, y_m, rcond=None)
            meta_loadings.append([float(coef_m[0]), float(coef_m[1])])
        except np.linalg.LinAlgError:
            meta_loadings.append([0.0, 0.0])

    result["meta_loadings"]    = meta_loadings     # (D_META, 2) loading on (x_h, x_r)
    result["meta_obs_counts"]  = meta_obs_counts   # how many obs per meta channel

    return result


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_stays(params: list[dict], M: int, seed: int = 42) -> np.ndarray:
    features = []
    for p in params:
        features.append([
            p["beta_hemo"],  p["beta_resp"],
            p["drift_hemo"], p["drift_resp"],
            p["alpha_hemo"], p["alpha_resp"],
        ])
    F = np.array(features, dtype=np.float64)
    F_norm = (F - F.mean(0)) / (F.std(0) + 1e-6)

    rng = np.random.default_rng(seed)
    centers = F_norm[rng.choice(len(F_norm), M, replace=False)]
    for _ in range(200):
        dists  = np.stack([np.sum((F_norm - c) ** 2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        new_c  = np.array([F_norm[labels == m].mean(0) if (labels == m).any()
                           else centers[m] for m in range(M)])
        if np.allclose(centers, new_c, atol=1e-6):
            break
        centers = new_c
    return labels


def compute_type_params(params: list[dict], labels: np.ndarray, M: int) -> dict:
    type_data: dict[int, list] = defaultdict(list)
    for i, p in enumerate(params):
        type_data[int(labels[i])].append(p)

    stats = {}
    for m in range(M):
        grp = type_data[m]
        if not grp:
            continue
        def med(key): return float(np.median([p[key] for p in grp]))
        stats[m] = {
            "count":      len(grp),
            "alpha_hemo": round(np.clip(med("alpha_hemo"), 0.40, 0.95), 4),
            "alpha_resp": round(np.clip(med("alpha_resp"), 0.40, 0.95), 4),
            "beta_hemo":  round(med("beta_hemo"),  4),
            "beta_resp":  round(med("beta_resp"),  4),
            "drift_hemo": round(med("drift_hemo"), 4),
            "drift_resp": round(med("drift_resp"), 4),
            "sigma_hemo": round(max(med("sigma_hemo"), 0.05), 4),
            "sigma_resp": round(max(med("sigma_resp"), 0.05), 4),
            "c_rh":       round(float(np.median([p["c_rh"] for p in grp])), 4),
            "c_hr":       round(float(np.median([p["c_hr"] for p in grp])), 4),
        }

    # Sort by (beta_hemo + beta_resp) ascending: type 0 = worst responder
    sorted_types = sorted(stats.keys(), key=lambda m: stats[m]["beta_hemo"] + stats[m]["beta_resp"])
    remap = {old: new for new, old in enumerate(sorted_types)}

    type_names = ["septic_shock", "resp_failure", "hemo_instability", "recovering"][:M]
    final = {}
    for old_m, new_m in remap.items():
        s = stats[old_m]
        final[str(new_m)] = {
            "name":       type_names[new_m] if new_m < len(type_names) else f"type_{new_m}",
            "count":      s["count"],
            "alpha_hemo": s["alpha_hemo"],
            "alpha_resp": s["alpha_resp"],
            "beta_hemo":  s["beta_hemo"],
            "beta_resp":  s["beta_resp"],
            "drift_hemo": s["drift_hemo"],
            "drift_resp": s["drift_resp"],
            "sigma_hemo": s["sigma_hemo"],
            "sigma_resp": s["sigma_resp"],
            "c_rh":       s["c_rh"],
            "c_hr":       s["c_hr"],
        }
    return final


def compute_population_loading(params: list[dict]) -> list:
    """Median loading matrix across stays."""
    Ls = np.array([p["loading_matrix"] for p in params])   # (n_stays, D_VITAL, 2)
    return np.median(Ls, axis=0).tolist()


def compute_meta_stats(params: list[dict]) -> dict:
    """Population-level meta loading and observation probability."""
    obs_counts = np.array([p["meta_obs_counts"] for p in params], dtype=np.float64)
    total_pairs = []
    for p in params:
        # rough estimate: obs_count / n_pairs
        total_pairs.append(max(p.get("meta_obs_counts", [0])[0], 1))

    # Meta loading: median of stays that had enough meta observations
    meta_loads = []
    for d in range(D_META):
        good = [p["meta_loadings"][d] for p in params
                if p["meta_obs_counts"][d] >= 3]
        if good:
            meta_loads.append(np.median(good, axis=0).tolist())
        else:
            meta_loads.append([0.5, 0.5])

    # Observation probability: fraction of steps where meta was present
    p_obs = []
    for d in range(D_META):
        counts = obs_counts[:, d]
        # approximate T per stay as 24 (1 day minimum)
        p_obs.append(float(np.clip(np.median(counts) / 24.0, 0.05, 0.95)))

    return {"meta_loadings": meta_loads, "p_obs_meta": p_obs}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../MIMIC/mimic_rmab_episodes_000000000000.csv")
    parser.add_argument("--out", default="mimic_params_v2.json")
    parser.add_argument("--M",   type=int, default=4)
    parser.add_argument("--max_stays", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    stays = load_stays(args.csv)
    print(f"  {len(stays)} ICU stays loaded")

    stay_ids = sorted(stays.keys())
    if args.max_stays > 0:
        stay_ids = stay_ids[:args.max_stays]

    print("Fitting 2D AR(1) per stay ...")
    fitted = []
    for i, sid in enumerate(stay_ids):
        if i % 2000 == 0:
            print(f"  {i}/{len(stay_ids)} ...", flush=True)
        r = fit_stay_2d(stays[sid])
        if r is not None:
            fitted.append(r)

    print(f"  {len(fitted)} stays with >= {MIN_HOURS}h complete vitals")

    print(f"Clustering into M={args.M} types ...")
    labels = cluster_stays(fitted, M=args.M)
    for m in range(args.M):
        print(f"  Type {m}: {(labels == m).sum()} stays")

    print("Computing type parameters ...")
    types = compute_type_params(fitted, labels, args.M)

    print("Computing population loading matrix ...")
    loading_matrix = compute_population_loading(fitted)

    print("Computing meta (temp/glucose) loading stats ...")
    meta_stats = compute_meta_stats(fitted)

    print("\n=== Fitted Type Parameters ===")
    for m in range(args.M):
        t = types[str(m)]
        print(f"  Type {m} ({t['name']}, n={t['count']}): "
              f"a_h={t['alpha_hemo']:.3f}  a_r={t['alpha_resp']:.3f}  "
              f"b_h={t['beta_hemo']:.3f}  b_r={t['beta_resp']:.3f}  "
              f"d_h={t['drift_hemo']:.3f}  d_r={t['drift_resp']:.3f}")

    print("\n=== Loading Matrix (5 vitals x 2 dims) ===")
    vital_names = ["HR", "SBP", "MBP", "SpO2", "RR"]
    for d, row in enumerate(loading_matrix):
        print(f"  {vital_names[d]:5s}: hemo={row[0]:.3f}  resp={row[1]:.3f}")

    output = {
        "obs_dim":    9,
        "state_dim":  3,
        "M":          args.M,
        "vital_names": VITAL_COLS,
        "meta_names":  META_COLS,
        "vital_mean":  VITAL_MEAN.tolist(),
        "vital_scale": VITAL_SCALE.tolist(),
        "vital_sign":  VITAL_SIGN.tolist(),
        "meta_mean":   META_MEAN.tolist(),
        "meta_scale":  META_SCALE.tolist(),
        "meta_sign":   META_SIGN.tolist(),
        "loading_matrix": loading_matrix,
        "meta_loadings":  meta_stats["meta_loadings"],
        "p_obs_meta":     meta_stats["p_obs_meta"],
        "types":          types,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
