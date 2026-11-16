"""Test whether research routes follow the actual treatment of each material family.

Input: family_eu_table.csv (build_family_tables.py). For the families with
Eurostat treatment data, the script reports:
  - Spearman correlation between the EU share of a family treated by a route
    and the share of papers on that family that study the route, for
    recycling and for thermal treatment, with a two-sided permutation p value
    and a percentile bootstrap interval (fixed seed);
  - the concentration of tonnes not recycled and of papers in the families
    that hold most of those tonnes;
  - the tonnage-weighted EU shares and the paper-weighted research shares.
It writes alignment_stats.csv and the figure fig_alignment.pdf / .png.

Usage:
    python analyze_alignment.py
"""
import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, N_PERM, N_BOOT, TOP = 42, 20000, 5000, 3
PAIRS = [("recycling", "eu_pct_recycling", "papers_pct_recycling"),
         ("thermal", "eu_pct_thermal", "papers_pct_thermal")]
LABELLED = {"recycling": {"other_mineral", "combustion_residues", "healthcare", "municipal_mixed",
                          "soils_dredging", "wood", "sludge", "chemical", "glass"},
            "thermal": {"healthcare", "municipal_mixed", "wood", "sludge", "chemical"}}
AXIS = {"recycling": "recycling", "thermal": "thermal treatment"}


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / var if var else float("nan")


def main():
    with (HERE / "family_eu_table.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k != "family":
                r[k] = float(v)
    rng = random.Random(SEED)
    stats = []
    print(f"families with Eurostat treatment data: {len(rows)}")
    for name, eu_key, res_key in PAIRS:
        a, b = [r[eu_key] for r in rows], [r[res_key] for r in rows]
        rho = spearman(a, b)
        perm = 0
        for _ in range(N_PERM):
            shuffled = b[:]
            rng.shuffle(shuffled)
            perm += abs(spearman(a, shuffled)) >= abs(rho) - 1e-12
        boots = []
        idx = list(range(len(rows)))
        for _ in range(N_BOOT):
            s = [rng.choice(idx) for _ in idx]
            value = spearman([a[i] for i in s], [b[i] for i in s])
            if not math.isnan(value):
                boots.append(value)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        p = (perm + 1) / (N_PERM + 1)
        stats.append({"route": name, "families": len(rows), "spearman_rho": round(rho, 4),
                      "permutation_p": round(p, 4), "bootstrap_ci_low": round(lo, 4),
                      "bootstrap_ci_high": round(hi, 4)})
        print(f"{name}: rho {rho:.4f} | permutation p {p:.4f} | bootstrap 95% CI "
              f"{lo:.4f} to {hi:.4f}")

    tonnes = sum(r["tonnes_treated"] for r in rows)
    papers = sum(r["papers"] for r in rows)
    for name, eu_key, res_key in PAIRS:
        eu = sum(r["tonnes_treated"] * r[eu_key] for r in rows) / tonnes
        res = sum(r["papers"] * r[res_key] for r in rows) / papers
        print(f"{name}: EU tonnage-weighted share {eu:.2f}% | research paper-weighted share "
              f"{res:.2f}%")
    not_rec = sum(r["tonnes_not_recycled"] for r in rows)
    top = sorted(rows, key=lambda r: -r["tonnes_not_recycled"])[:TOP]
    top_nr = sum(r["tonnes_not_recycled"] for r in top)
    top_p = sum(r["papers"] for r in top)
    print(f"top {TOP} families by tonnes not recycled ({', '.join(r['family'] for r in top)}): "
          f"{100 * top_nr / not_rec:.2f}% of tonnes not recycled, {top_p:.0f} of {papers:.0f} "
          f"papers ({100 * top_p / papers:.2f}%)")
    stats.append({"route": f"top{TOP}_not_recycled", "families": TOP,
                  "share_tonnes_not_recycled": round(100 * top_nr / not_rec, 2),
                  "share_papers": round(100 * top_p / papers, 2)})
    fields = sorted({k for s in stats for k in s}, key=lambda k: (k != "route", k))
    with (HERE / "alignment_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5), constrained_layout=True)
    titles = {"recycling": "Recycling", "thermal": "Thermal treatment (energy recovery + incineration)"}
    for ax, (name, eu_key, res_key), st in zip(axes, PAIRS, stats):
        ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1, ls="--", zorder=1)
        ax.scatter([r[eu_key] for r in rows], [r[res_key] for r in rows], s=28,
                   color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.2, zorder=2)
        for r in rows:
            if r["family"] in LABELLED[name]:
                ax.annotate(r["family"].replace("_", " "), (r[eu_key], r[res_key]),
                            xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                            color="#52514e")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel(f"EU-27 waste treated by {AXIS[name]} in 2022 (%)")
        ax.set_ylabel(f"Papers studying {AXIS[name]} (%)")
        ax.set_title(f"{titles[name]}\nSpearman \u03c1 = {st['spearman_rho']:.4f}, "
                     f"p = {st['permutation_p']:.4f}", fontsize=8, color="#0b0b0b")
        ax.grid(color="#ebeae6", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_alignment.{ext}", dpi=300)
    print("wrote alignment_stats.csv and fig_alignment.pdf / .png")


if __name__ == "__main__":
    main()
