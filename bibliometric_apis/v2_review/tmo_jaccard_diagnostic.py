"""
=========================

Pair-wise AWS diagnostic: Jaccard overlap of TMO terms between cited papers
vs. random pairs, separated by axis (Techniques, Materials, Objectives).

This script complements the cluster-level CSC-score by providing pair-level
evidence: even verified cross-cluster citations show very low TMO overlap,
quantifying AWS at the granularity of individual citation pairs.

Outputs:
  runs/paper_tmo_retrieval_v2/jaccard_pair_diagnostic.json
  runs/paper_tmo_retrieval_v2/jaccard_pair_per_pair.csv

Usage:
  python tmo_jaccard_diagnostic.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TMO_PATH    = Path("runs/paper_tmo/paper_tmo.csv")
EDGES_PATH  = Path("backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv")
OUT_DIR     = Path("runs/paper_tmo_retrieval_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42


def terms(cell: str) -> set[str]:
    """Extract unique word tokens (len >=3) from a pipe-separated TMO cell."""
    if not isinstance(cell, str) or not cell.strip():
        return set()
    out: set[str] = set()
    for phrase in cell.split("|"):
        for w in re.findall(r"[a-z]+", phrase.lower()):
            if len(w) >= 3:
                out.add(w)
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    print("Loading TMO data...")
    tmo = pd.read_csv(TMO_PATH)
    tmo = tmo[tmo["status"] == "ok"]
    print(f"  {len(tmo)} papers with TMO")

    T = {r["doi"]: terms(r.get("techniques", "")) for _, r in tmo.iterrows()}
    M = {r["doi"]: terms(r.get("materials",  "")) for _, r in tmo.iterrows()}
    O = {r["doi"]: terms(r.get("objectives", "")) for _, r in tmo.iterrows()}

    edges = pd.read_csv(EDGES_PATH)
    print(f"  {len(edges)} cross-cluster citation pairs")

    # Cited pairs (verified, both have TMO)
    cited_rows = []
    per_pair_rows = []
    for _, e in edges.iterrows():
        s, t = e["source_doi"], e["target_doi"]
        if s in T and t in T:
            t_jac = jaccard(T[s], T[t])
            m_jac = jaccard(M[s], M[t])
            o_jac = jaccard(O[s], O[t])
            cited_rows.append((t_jac, m_jac, o_jac))
            per_pair_rows.append({
                "source_doi": s, "target_doi": t,
                "jaccard_T": round(t_jac, 4),
                "jaccard_M": round(m_jac, 4),
                "jaccard_O": round(o_jac, 4),
                "max_jaccard": round(max(t_jac, m_jac, o_jac), 4),
                "zero_overlap_all_axes": int(max(t_jac, m_jac, o_jac) == 0),
            })

    arr = np.array(cited_rows)
    n = len(arr)
    print(f"\n  Cited pairs with TMO in both papers: {n}")

    # Statistics on cited pairs
    cited_mean_t = float(arr[:, 0].mean())
    cited_mean_m = float(arr[:, 1].mean())
    cited_mean_o = float(arr[:, 2].mean())

    best_axis = arr.argmax(axis=1)
    win_t = int((best_axis == 0).sum())
    win_m = int((best_axis == 1).sum())
    win_o = int((best_axis == 2).sum())

    max_ov = arr.max(axis=1)
    cited_mean_max = float(max_ov.mean())
    zero_all = int((max_ov == 0).sum())

    # Random control (same n)
    all_dois = list(T.keys())
    rng = np.random.default_rng(RNG_SEED)
    rand_rows = []
    for _ in range(n):
        s, t = rng.choice(all_dois, 2, replace=False)
        rand_rows.append((jaccard(T[s], T[t]),
                          jaccard(M[s], M[t]),
                          jaccard(O[s], O[t])))
    rand_arr = np.array(rand_rows)

    rand_mean_t = float(rand_arr[:, 0].mean())
    rand_mean_m = float(rand_arr[:, 1].mean())
    rand_mean_o = float(rand_arr[:, 2].mean())
    rand_mean_max = float(rand_arr.max(axis=1).mean())

    # Print summary
    print("\n=== AWS PAIR-WISE DIAGNOSTIC (Jaccard TMO) ===")
    print(f"\nCITED pairs (n={n}):")
    print(f"  Technique : {cited_mean_t:.3f}")
    print(f"  Material  : {cited_mean_m:.3f}")
    print(f"  Objective : {cited_mean_o:.3f}")
    print(f"  Max-axis  : {cited_mean_max:.3f}")
    print(f"\nWinning axis per pair:")
    print(f"  Technique: {win_t} ({100*win_t/n:.0f}%)")
    print(f"  Material : {win_m} ({100*win_m/n:.0f}%)")
    print(f"  Objective: {win_o} ({100*win_o/n:.0f}%)")
    print(f"\nPairs with ZERO overlap in all 3 axes: {zero_all} "
          f"({100*zero_all/n:.0f}%)")
    print(f"\nRANDOM control (n={n}):")
    print(f"  Technique : {rand_mean_t:.3f}")
    print(f"  Material  : {rand_mean_m:.3f}")
    print(f"  Objective : {rand_mean_o:.3f}")
    print(f"  Max-axis  : {rand_mean_max:.3f}")
    print(f"\nLift (cited - random):")
    print(f"  Technique : +{cited_mean_t - rand_mean_t:.3f} "
          f"({cited_mean_t/max(rand_mean_t,1e-9):.1f}x)")
    print(f"  Material  : +{cited_mean_m - rand_mean_m:.3f} "
          f"({cited_mean_m/max(rand_mean_m,1e-9):.1f}x)")
    print(f"  Objective : +{cited_mean_o - rand_mean_o:.3f} "
          f"({cited_mean_o/max(rand_mean_o,1e-9):.1f}x)")
    print(f"  Max-axis  : +{cited_mean_max - rand_mean_max:.3f} "
          f"({cited_mean_max/max(rand_mean_max,1e-9):.1f}x)")

    # Save outputs
    summary = {
        "n_cited_pairs": n,
        "cited": {
            "technique_mean": round(cited_mean_t, 4),
            "material_mean":  round(cited_mean_m, 4),
            "objective_mean": round(cited_mean_o, 4),
            "max_axis_mean":  round(cited_mean_max, 4),
            "n_zero_overlap_all_axes": zero_all,
            "pct_zero_overlap_all_axes": round(100 * zero_all / n, 1),
            "winning_axis_counts": {"T": win_t, "M": win_m, "O": win_o},
        },
        "random_control": {
            "technique_mean": round(rand_mean_t, 4),
            "material_mean":  round(rand_mean_m, 4),
            "objective_mean": round(rand_mean_o, 4),
            "max_axis_mean":  round(rand_mean_max, 4),
            "seed": RNG_SEED,
        },
        "lift_cited_vs_random": {
            "technique": round(cited_mean_t - rand_mean_t, 4),
            "material":  round(cited_mean_m - rand_mean_m, 4),
            "objective": round(cited_mean_o - rand_mean_o, 4),
            "max_axis":  round(cited_mean_max - rand_mean_max, 4),
        },
    }
    (OUT_DIR / "jaccard_pair_diagnostic.json").write_text(
        json.dumps(summary, indent=2))
    pd.DataFrame(per_pair_rows).to_csv(
        OUT_DIR / "jaccard_pair_per_pair.csv", index=False)

    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
