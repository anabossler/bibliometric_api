"""
VERSION DINAMICA de 09_chi2_northsouth.py — sin numeros hardcodeados.

Lee los conteos North/South directamente de geography_global_split_v3.csv
(output de 02_ce_geography_v3.py) y calcula chi2, p, dof, Cramér's V y
residuos estandarizados post-hoc. Genera tambien el snippet LaTeX listo.

Esto hace el resultado 100% reproducible: cualquiera que regenere la geografia
obtiene exactamente el mismo test, sin riesgo de que el paper y los datos
diverjan.

Run:
  python 09_chi2_northsouth_v3.py \
      --split results/ce_review_v3/geography_global_split_v3.csv \
      --out results/ce_review_v3/chi2_snippet.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(chi2: float, n: float, k: int, r: int) -> float:
    return float(np.sqrt(chi2 / (n * (min(k, r) - 1))))


def standardised_residuals(observed, expected):
    return (observed - expected) / np.sqrt(expected)


def load_observed(split_path: Path):
    """
    Lee geography_global_split_v3.csv y devuelve:
      observed = matriz [n_areas x 2] con columnas [North, South]
      areas    = lista de nombres de mega-area (en el orden de las filas)
    El CSV tiene columnas: mega_area, Global North, Global South, total, pct_north, pct_south
    """
    df = pd.read_csv(split_path)
    # localizar columnas de forma robusta
    cols = {c.lower().strip(): c for c in df.columns}
    area_col = next((cols[c] for c in cols if "mega" in c or "area" in c), df.columns[0])
    north_col = next((cols[c] for c in cols if "north" in c and "pct" not in c), None)
    south_col = next((cols[c] for c in cols if "south" in c and "pct" not in c), None)
    if north_col is None or south_col is None:
        sys.exit(f"ERROR: no encuentro columnas North/South en {split_path}. "
                 f"Columnas: {list(df.columns)}")
    df = df[[area_col, north_col, south_col]].copy()
    df.columns = ["mega_area", "north", "south"]
    df = df.dropna()
    observed = df[["north", "south"]].to_numpy(dtype=float)
    areas = df["mega_area"].tolist()
    return observed, areas


def run_test(observed, areas):
    chi2, p, dof, expected = chi2_contingency(observed, correction=False)
    n = observed.sum()
    v = cramers_v(chi2, n, observed.shape[1], observed.shape[0])
    sr = standardised_residuals(observed, expected)
    return {"chi2": chi2, "p": p, "dof": dof, "n": n, "cramers_v": v,
            "expected": expected, "residuals": sr, "mega_areas": areas}


def print_results(res):
    print("=" * 64)
    print("Chi-square: mega-area x North/South  (datos dinamicos, v3)")
    print("=" * 64)
    print(f"  chi2({res['dof']}) = {res['chi2']:.3f}")
    print(f"  p-value           = {res['p']:.3e}")
    print(f"  N (paper x country units) = {int(res['n']):,}")
    print(f"  Cramér's V        = {res['cramers_v']:.3f}")
    print()
    print("Residuos estandarizados (>|1.96| = sig. al 5%):")
    print(f"  {'Mega-area':<42} {'North':>8}  {'South':>8}")
    print("  " + "-" * 62)
    for i, area in enumerate(res["mega_areas"]):
        rn, rs = res["residuals"][i]
        fn = " *" if abs(rn) > 1.96 else "  "
        fs = " *" if abs(rs) > 1.96 else "  "
        print(f"  {area:<42} {rn:+7.2f}{fn}  {rs:+7.2f}{fs}")
    print()
    v = res["cramers_v"] # Conventional interpretation thresholds for Cramér's V
    effect = ("negligible" if v < 0.1 else "small" if v < 0.2
              else "moderate" if v < 0.3 else "large")
    sig = ("p < 0.001" if res["p"] < 0.001 else f"p = {res['p']:.3f}")
    print(f"  Interpretacion: chi2 ({sig}), efecto {effect} (V={v:.3f}).")
    print()


def latex_snippet(res):
    v, p, chi2, dof = res["cramers_v"], res["p"], res["chi2"], res["dof"]
    p_str = r"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    lines = [
        r"% ---- AUTO-GENERADO por 09_chi2_northsouth_v3.py (no editar a mano) ----",
        r"The association between mega-area and Global North/South authorship",
        r"is statistically significant: $\chi^2(" + str(dof) + r") = "
        + f"{chi2:.2f}$, ${p_str}$, "
        + r"Cram\'{e}r's $V = " + f"{v:.3f}$.",
    ]
    notable = []
    for i, area in enumerate(res["mega_areas"]):
        for j, region in enumerate(["Global North", "Global South"]):
            sr = res["residuals"][i, j]
            if abs(sr) > 1.96:
                d = "over-represented" if sr > 0 else "under-represented"
                notable.append(f"{region} in {area} ({d}, $z = {sr:+.2f}$)")
    if notable:
        lines.append(r"Post-hoc standardised residuals identify the following")
        lines.append(r"cells as driving the association: " + "; ".join(notable) + ".")
    lines.append(r"% ---- end ----")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    help="geography_global_split_v3.csv (output del v3)")
    ap.add_argument("--out", default=None, help="opcional: escribe snippet LaTeX aqui")
    args = ap.parse_args()

    observed, areas = load_observed(Path(args.split))
    print(f"Cargados {int(observed.sum()):,} paper-country units desde {args.split}\n")
    res = run_test(observed, areas)
    print_results(res)
    snip = latex_snippet(res)
    print("LaTeX snippet:")
    print("-" * 64); print(snip); print("-" * 64)
    if args.out:
        Path(args.out).write_text(snip, encoding="utf-8")
        print(f"\nSnippet escrito en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
