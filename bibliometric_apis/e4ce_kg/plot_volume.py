"""Figure 3: share of papers and share of tonnes treated by material family.

Reads family_eu_table.csv (written by build_family_tables.py) and draws, for each
of the 22 families, its share of the papers and its share of the EU-27 tonnes
treated in 2022 on a logarithmic axis, with the families ordered by tonnes
treated. The three families with the most tonnes not recycled are set in bold.

Usage:
    python plot_volume.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = 3
NAMES = {
    "other_mineral": "Other mineral", "soils_dredging": "Soils and dredging spoils",
    "municipal_mixed": "Mixed municipal", "construction_demolition": "Construction and demolition",
    "combustion_residues": "Combustion residues", "wood": "Wood", "sludge": "Sludges",
    "chemical": "Chemical", "plastic": "Plastic", "vegetal_waste": "Vegetal", "food_waste": "Food",
    "manure": "Manure", "healthcare": "Healthcare", "rubber": "Rubber",
    "paper_cardboard": "Paper and cardboard", "textile": "Textile", "used_oils": "Used oils",
    "glass": "Glass", "metal": "Metal", "weee": "WEEE", "batteries": "Batteries",
    "vehicles": "End-of-life vehicles",
}
PAPERS_COLOR, TONNES_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#ebeae6", "#fcfcfb"


def main():
    with (HERE / "family_eu_table.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tonnes = {r["family"]: float(r["tonnes_treated"]) for r in rows}
    not_rec = {r["family"]: float(r["tonnes_not_recycled"]) for r in rows}
    papers = {r["family"]: int(r["papers"]) for r in rows}
    missing = set(tonnes) - set(NAMES)
    if missing:
        raise SystemExit(f"families without a display name: {sorted(missing)}")
    tt, tp, tn = sum(tonnes.values()), sum(papers.values()), sum(not_rec.values())
    share_t = {f: 100 * tonnes[f] / tt for f in tonnes}
    share_p = {f: 100 * papers[f] / tp for f in tonnes}
    top = sorted(tonnes, key=lambda f: -not_rec[f])[:TOP]
    print(f"{len(tonnes)} families | {tp} papers | {tt / 1e6:.2f} Mt treated")
    print(f"top {TOP} by tonnes not recycled {top}: "
          f"{100 * sum(not_rec[f] for f in top) / tn:.2f}% of tonnes not recycled, "
          f"{100 * sum(papers[f] for f in top) / tp:.2f}% of papers")
    for f in sorted(tonnes, key=lambda x: -tonnes[x]):
        print(f"  {f} | {share_p[f]:.2f}% of papers | {share_t[f]:.2f}% of tonnes treated")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    order = sorted(tonnes, key=lambda f: tonnes[f])
    fig, ax = plt.subplots(figsize=(7.48, 4.6), constrained_layout=True)
    for y, f in enumerate(order):
        ax.plot([share_p[f], share_t[f]], [y, y], color="#c3c2b7", lw=1, zorder=1)
    ys = range(len(order))
    ax.scatter([share_p[f] for f in order], ys, s=30, marker="o", color=PAPERS_COLOR,
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Share of the 9,956 papers")
    ax.scatter([share_t[f] for f in order], ys, s=28, marker="s", color=TONNES_COLOR,
               edgecolor=SURFACE, linewidth=1.2, zorder=2,
               label="Share of the EU-27 tonnes treated, 2022")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES[f] for f in order])
    for label, f in zip(ax.get_yticklabels(), order):
        label.set_color(INK)
        if f in top:
            label.set_fontweight("bold")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 60)
    ticks = [0.05, 0.1, 0.5, 1, 5, 10, 50]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Share of the 22 families (%, logarithmic scale)")
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_volume.{ext}", dpi=300)
    print("wrote fig_volume.pdf / .png")


if __name__ == "__main__":
    main()
