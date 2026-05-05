"""
Chi-square test of independence between mega-area and Global North/South region.
Uses the counts from Table 4 (tab:northsouth) of the CE review paper.

Usage:
    python chi2_northsouth.py

Usage (raw data):
    python chi2_northsouth.py --affiliations ./results/ce_review/paper_country.csv

Output:
    - chi2 statistic, p-value, degrees of freedom, Cramér's V
    - standardised residuals per cell (post-hoc)
    - LaTeX snippet ready to paste into the paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Published counts from Table 4 (tab:northsouth)
# Rows = mega-areas, columns = [North, South]
# ---------------------------------------------------------------------------

MEGA_AREAS = [
    "Industrial Sectors & Applied Cases",
    "Business, Policy & Governance",
    "Energy & Resource Systems",
    "Materials & Technical Recycling",
    "Sustainability Framing & Society",
]

# counts at the paper x country level (a paper with multi-country authors
# is counted once per country, so totals exceed n_papers)
OBSERVED = np.array([
    [4430, 1757],   # Industrial Sectors
    [5824, 2414],   # Business, Policy
    [3410, 1505],   # Energy
    [2164, 1106],   # Materials
    [3631, 2028],   # Sustainability Framing
], dtype=float)


def cramers_v(chi2: float, n: float, k: int, r: int) -> float:
    """Cramér's V effect size. k = n_cols, r = n_rows."""
    return np.sqrt(chi2 / (n * (min(k, r) - 1)))


def standardised_residuals(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (observed - expected) / np.sqrt(expected)


def run_test(observed: np.ndarray, mega_areas: list[str]) -> dict:
    chi2, p, dof, expected = chi2_contingency(observed, correction=False)
    n = observed.sum()
    v = cramers_v(chi2, n, observed.shape[1], observed.shape[0])
    sr = standardised_residuals(observed, expected)
    return {
        "chi2": chi2, "p": p, "dof": dof, "n": n,
        "cramers_v": v, "expected": expected, "residuals": sr,
        "mega_areas": mega_areas,
    }


def print_results(res: dict) -> None:
    print("=" * 60)
    print("Chi-square test of independence: mega-area x North/South")
    print("=" * 60)
    print(f"  chi2({res['dof']}) = {res['chi2']:.3f}")
    print(f"  p-value           = {res['p']:.2e}")
    print(f"  N (paper x country units) = {int(res['n']):,}")
    print(f"  Cramér's V        = {res['cramers_v']:.3f}")
    print()

    print("Standardised residuals (>|1.96| = significant at 5%):")
    print(f"  {'Mega-area':<42} {'North':>8}  {'South':>8}")
    print("  " + "-" * 62)
    for i, area in enumerate(res["mega_areas"]):
        rn, rs = res["residuals"][i]
        flag_n = " *" if abs(rn) > 1.96 else "  "
        flag_s = " *" if abs(rs) > 1.96 else "  "
        print(f"  {area:<42} {rn:+7.2f}{flag_n}  {rs:+7.2f}{flag_s}")
    print()

    # Interpretation
    if res["p"] < 0.001:
        sig = "p < 0.001"
    elif res["p"] < 0.01:
        sig = f"p = {res['p']:.3f}"
    else:
        sig = f"p = {res['p']:.2f}"

    if res["cramers_v"] < 0.1:
        effect = "negligible"
    elif res["cramers_v"] < 0.2:
        effect = "small"
    elif res["cramers_v"] < 0.3:
        effect = "moderate"
    else:
        effect = "large"

    print(f"  Interpretation: chi2 test is significant ({sig}),")
    print(f"  effect size is {effect} (V = {res['cramers_v']:.3f}).")
    print()


def latex_snippet(res: dict) -> str:
    v = res["cramers_v"]
    p = res["p"]
    chi2 = res["chi2"]
    dof = res["dof"]

    p_str = r"p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    lines = [
        r"% ---- paste into Section 6 (Geography) ----",
        r"The association between mega-area and Global North/South authorship",
        r"is statistically significant: $\chi^2(" + str(dof) + r") = "
        + f"{chi2:.2f}$, ${p_str}$, "
        + r"Cram\'{e}r's $V = " + f"{v:.3f}$.",
    ]

    # residuals worth mentioning (|sr| > 1.96)
    notable = []
    for i, area in enumerate(res["mega_areas"]):
        for j, region in enumerate(["Global North", "Global South"]):
            sr = res["residuals"][i, j]
            if abs(sr) > 1.96:
                direction = "over-represented" if sr > 0 else "under-represented"
                notable.append(f"{region} in {area} ({direction}, "
                               f"$z = {sr:+.2f}$)")

    if notable:
        lines.append(r"Post-hoc standardised residuals identify the following")
        lines.append(r"cells as driving the association: " + "; ".join(notable) + ".")

    lines.append(r"% ---- end paste ----")
    return "\n".join(lines)


def from_raw_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Build contingency table from a raw paper-country CSV.
    Expected columns: mega_area, region  (region in {North, South} or {Global North, Global South})
    """
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise region column
    region_col = next((c for c in df.columns if "region" in c or "north" in c.lower()), None)
    area_col   = next((c for c in df.columns if "mega" in c or "area" in c), None)

    if region_col is None or area_col is None:
        sys.exit(f"ERROR: CSV must have columns for mega_area and region. "
                 f"Found: {list(df.columns)}")

    df["_north"] = df[region_col].str.lower().str.contains("north").astype(int)
    ct = df.groupby([area_col, "_north"]).size().unstack(fill_value=0)
    areas = list(ct.index)
    obs = ct.values.astype(float)
    # ensure columns are [South=0, North=1]
    if obs.shape[1] == 2:
        obs = obs[:, [1, 0]]  # -> [North, South]
    return obs, areas


def main() -> int:
    ap = argparse.ArgumentParser(description="Chi-square North-South test for CE review paper.")
    ap.add_argument("--affiliations", default=None,
                    help="Optional: path to raw paper-country CSV with columns "
                         "mega_area, region. If omitted, uses published Table 4 counts.")
    ap.add_argument("--out", default=None,
                    help="Optional: write LaTeX snippet to this file.")
    args = ap.parse_args()

    if args.affiliations:
        obs, areas = from_raw_csv(Path(args.affiliations))
        print(f"Loaded {int(obs.sum()):,} paper-country units from {args.affiliations}")
    else:
        obs, areas = OBSERVED, MEGA_AREAS
        print("Using published counts from Table 4 (tab:northsouth).")

    res = run_test(obs, areas)
    print_results(res)

    snippet = latex_snippet(res)
    print("LaTeX snippet:")
    print("-" * 60)
    print(snippet)
    print("-" * 60)

    if args.out:
        Path(args.out).write_text(snippet, encoding="utf-8")
        print(f"\nSnippet written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
