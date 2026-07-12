"""
=================

Reconstruct an exploratory adafina / cocido / olla lineage tree from the RELISH
recipe graph.

The script selects a one-pot boiled-stew lineage across:

  - Sephardic/Maghreb dafina-like recipes
  - Salonika/Balkan Sephardic variants
  - Catalan escudella / olla cousins
  - medieval Iberian potaje / olla anchors

It deliberately excludes obvious non-stew Sephardic dishes such as eggplant
dishes, pastries, salads, and desserts so the tree focuses on the cocido /
adafina lineage rather than the full Sephardic clade.

The resulting tree is a similarity tree, not proof of culinary descent. It is
an exploratory phylogenetic analogy built from recipe character profiles.

Default input
-------------

  data/graph_step4_foodon.gpickle

Default outputs
---------------

  data/phylo_adafina.nwk
  data/phylo_adafina.png
  data/phylo_adafina_taxa.csv
  data/phylo_adafina_summary.json

Dependencies
------------

  recipe_phylogenetics.py
  biopython
  matplotlib
  numpy
  pandas
  networkx

Usage
-----

  python step_phylogeny.py

  python step_phylogeny.py \
    --graph data/graph_step4_foodon.gpickle \
    --out-prefix data/phylo_adafina

  python step_phylogeny.py --keep-all-sefardi

  python step_phylogeny.py --list

  python step_phylogeny.py --skip-figure

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import recipe_phylogenetics as rp


DEFAULT_GRAPH = Path("data/graph_step4_foodon.gpickle")
DEFAULT_OUT_PREFIX = Path("data/phylo_adafina")

SEFARDI_SRC = "sefardies_es"
CATALAN_SRC = "cuina_catalana"
IBERIAN_MEDIEVAL_SRC = {"a_miscellany", "enseignements"}

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step_phylogeny")


# ---------------------------------------------------------------------------
# Lineage definition
# ---------------------------------------------------------------------------

STEW_TITLE_RE = re.compile(
    r"\b("
    r"adafina|dafina|msoki|m'soki|kodrero|almoronia|almoron[ií]a|"
    r"guisado|guisad|estofado|cocido|escudella|olla|potaje|puchero|"
    r"hervido|cozido|hamin|hom[ií]n|schena|sc[eh]ina"
    r")\b",
    re.IGNORECASE,
)

EXCLUDE_TITLE_RE = re.compile(
    r"\b("
    r"berenjena|almodrote|pepitada|gazpacho|ensalada|torta|tarta|"
    r"galleta|orejas|haman|januca|hanuka|shavuot|purim|bizcocho|"
    r"pud[ií]n|pastel|rosquilla|jala|jalot|gato|suljaniot|spaguetti|"
    r"empanada|sandwich|carpaccio|lasana|tatin|crema|pollo|arroz con|"
    r"quesadilla|coliflor|alcachofa|granada|musaka"
    r")\b",
    re.IGNORECASE,
)


BRANCH_COLORS = {
    "Maghreb (dafina)": "#d1495b",
    "Balkans / Salonika": "#3a7ca5",
    "Andalusi root": "#8b5a2b",
    "Catalan cocido/olla": "#e3a72f",
    "Sephardic (other)": "#6a4c93",
    "Medieval Iberian": "#2a9d8f",
    "Other": "#444444",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def branch_of(meta: dict[str, Any]) -> tuple[str, str]:
    """Map a recipe to a geographic branch label and color."""
    place = safe_str(meta.get("place")).lower()
    source_id = meta.get("source_id")

    if any(keyword in place for keyword in ("morocco", "maghreb", "tunis", "tunisia")):
        label = "Maghreb (dafina)"
    elif any(keyword in place for keyword in ("salonika", "salonica", "rhodes", "greece", "ottoman", "balkan")):
        label = "Balkans / Salonika"
    elif any(keyword in place for keyword in ("andalus", "andalusi", "iberian", "al-andalus")):
        label = "Andalusi root"
    elif source_id == CATALAN_SRC:
        label = "Catalan cocido/olla"
    elif source_id == SEFARDI_SRC:
        label = "Sephardic (other)"
    elif source_id in IBERIAN_MEDIEVAL_SRC:
        label = "Medieval Iberian"
    else:
        label = "Other"

    return label, BRANCH_COLORS[label]


def is_stew_title(title: str) -> bool:
    return bool(STEW_TITLE_RE.search(title or ""))


def is_excluded_non_stew(title: str) -> bool:
    return bool(EXCLUDE_TITLE_RE.search(title or ""))


def select_lineage_nodes(
    graph: Any,
    *,
    keep_all_sefardi: bool,
    include_sources: set[str],
    exclude_sources: set[str],
) -> set[str]:
    """Return Recipe node ids selected for the adafina/cocido lineage."""
    selected: set[str] = set()

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        source_id = attrs.get("source_id")
        title = safe_str(attrs.get("title"))

        if source_id in exclude_sources:
            continue

        if include_sources and source_id not in include_sources:
            continue

        if source_id == SEFARDI_SRC:
            if keep_all_sefardi:
                selected.add(str(node_id))
            elif is_stew_title(title) and not is_excluded_non_stew(title):
                selected.add(str(node_id))

        elif source_id == CATALAN_SRC:
            if is_stew_title(title):
                selected.add(str(node_id))

        elif source_id in IBERIAN_MEDIEVAL_SRC:
            if is_stew_title(title):
                selected.add(str(node_id))

    return selected


def load_extra_nodes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Extra node file not found: {path}")

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def feature_kind_set(values: list[str]) -> set[str]:
    if values:
        return set(values)
    return {"ingredient", "foodon", "action"}


def make_tip_label(index: int, meta: dict[str, Any]) -> str:
    """Match recipe_phylogenetics.make_tip_label if available."""
    if hasattr(rp, "make_tip_label"):
        return rp.make_tip_label(index, meta)

    year = meta.get("year") if meta.get("year") is not None else "?"
    title = re.sub(r"[^A-Za-z0-9]+", "_", safe_str(meta.get("title"), "untitled")[:32]).strip("_")
    source = re.sub(r"[^A-Za-z0-9]+", "_", safe_str(meta.get("source_id"), "source"))[:20]
    return f"t{index:03d}_{title or 'untitled'}_{source}_{year}"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_by_branch(
    tree: Any,
    taxa: list[tuple[str, dict[str, Any], set[str]]],
    labels: list[str],
    out_png: Path,
    *,
    title: str,
) -> None:
    """Draw tree tips colored by geographic branch."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from Bio import Phylo
    except ImportError as exc:
        raise RuntimeError("matplotlib and Biopython are required for drawing") from exc

    meta_by_label: dict[str, tuple[str, str]] = {}

    for label, (_, meta, _) in zip(labels, taxa):
        meta_by_label[label] = branch_of(meta)

    def label_func(clade):
        return clade.name if clade.is_terminal() else ""

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, max(6, 0.34 * len(taxa))))
    Phylo.draw(tree, axes=ax, do_show=False, label_func=label_func)

    for text in ax.texts:
        raw = text.get_text().strip()
        if raw not in meta_by_label:
            continue

        branch_label, color = meta_by_label[raw]
        text.set_color(color)
        text.set_fontsize(7.5)

    ax.set_title(title, fontsize=12)

    seen: dict[str, str] = {}
    for branch_label, color in meta_by_label.values():
        seen[branch_label] = color

    handles = [mpatches.Patch(color=color, label=branch_label) for branch_label, color in seen.items()]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Wrote tree figure: %s", out_png)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def taxa_dataframe(
    taxa: list[tuple[str, dict[str, Any], set[str]]],
    labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, (recipe_id, meta, features) in zip(labels, taxa):
        branch_label, color = branch_of(meta)
        rows.append({
            "taxon_label": label,
            "recipe_id": recipe_id,
            "branch": branch_label,
            "branch_color": color,
            "title": meta.get("title"),
            "source_id": meta.get("source_id"),
            "source_title": meta.get("source_title"),
            "source_author": meta.get("source_author"),
            "year": meta.get("year"),
            "raw_year": meta.get("raw_year"),
            "period": meta.get("period"),
            "place": meta.get("place"),
            "language": meta.get("lang"),
            "n_characters": len(features),
        })

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def print_taxa_list(taxa: list[tuple[str, dict[str, Any], set[str]]]) -> None:
    for _, meta, features in sorted(taxa, key=lambda row: (branch_of(row[1])[0], safe_str(row[1].get("title")))):
        branch_label, _ = branch_of(meta)
        year = meta.get("year") if meta.get("year") is not None else "?"
        title = safe_str(meta.get("title"), "untitled")
        print(f"  [{branch_label:<22}] {title[:60]:60s} {year!s:>6s} ({len(features)} chars)")


def print_introgression_summary(introgression: dict[str, Any]) -> None:
    print("\nNew World carriers by period, Columbian-introgression signal:")
    for row in introgression.get("by_period", []):
        print(
            f"  {row['period']:15s} "
            f"{row['n_carriers']:3d}/{row['n_taxa']:<3d} "
            f"({row['carrier_pct']:5.1f}%)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    graph = rp.load_graph(args.graph)

    include_sources = set(args.include_source or [])
    exclude_sources = set(args.exclude_source or [])

    selected_nodes = select_lineage_nodes(
        graph,
        keep_all_sefardi=args.keep_all_sefardi,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
    )

    selected_nodes.update(args.extra_node or [])
    selected_nodes.update(load_extra_nodes(args.extra_node_file))

    log.info("Selected %d candidate recipes for the lineage", len(selected_nodes))

    taxa = rp.select_taxa(
        graph,
        extra_nodes=selected_nodes,
        min_features=args.min_features,
        cap=None if args.cap == 0 else args.cap,
        feature_kinds=feature_kind_set(args.feature_kind or []),
    )

    log.info("%d taxa after feature filter", len(taxa))

    if args.list:
        print_taxa_list(taxa)
        return 0

    if len(taxa) < 4:
        log.error("Too few taxa for a tree: %d. Loosen selection or lower --min-features.", len(taxa))
        return 1

    labels, characters, matrix = rp.build_character_matrix(taxa)
    distances = rp.distance_matrix(matrix)
    tree = rp.build_tree(labels, distances, taxa=taxa)

    newick_path = Path(f"{args.out_prefix}.nwk")
    png_path = Path(f"{args.out_prefix}.png")
    taxa_path = Path(f"{args.out_prefix}_taxa.csv")
    summary_path = Path(f"{args.out_prefix}_summary.json")
    introgression_path = Path(f"{args.out_prefix}_introgression.json")

    rp.export_newick(tree, newick_path)

    if not args.skip_figure:
        draw_by_branch(
            tree,
            taxa,
            labels,
            png_path,
            title=(
                f"Adafina / cocido lineage — {len(taxa)} taxa "
                f"(Sephardic + Catalan + Andalusi)"
            ),
        )

    markers = rp.load_new_world_markers(args.new_world_markers) if hasattr(rp, "load_new_world_markers") else set()
    if hasattr(rp, "map_introgression"):
        try:
            introgression = rp.map_introgression(taxa, markers=markers)
        except TypeError:
            per_taxon, carriers_by_period, totals_by_period = rp.map_introgression(taxa)
            introgression = {
                "per_taxon": per_taxon,
                "by_period": [
                    {
                        "period": period,
                        "n_taxa": int(totals_by_period[period]),
                        "n_carriers": int(carriers_by_period.get(period, 0)),
                        "carrier_pct": (
                            round(100 * carriers_by_period.get(period, 0) / totals_by_period[period], 2)
                            if totals_by_period[period]
                            else 0.0
                        ),
                    }
                    for period in sorted(totals_by_period)
                ],
            }
    else:
        introgression = {"per_taxon": [], "by_period": []}

    write_csv(taxa_dataframe(taxa, labels), taxa_path)
    write_json(introgression_path, introgression)

    branch_counts = Counter(branch_of(meta)[0] for _, meta, _ in taxa)

    summary = {
        "graph": str(args.graph),
        "out_prefix": str(args.out_prefix),
        "n_candidate_recipes": len(selected_nodes),
        "n_taxa": len(taxa),
        "n_characters": len(characters),
        "branch_counts": dict(branch_counts),
        "parameters": {
            "keep_all_sefardi": args.keep_all_sefardi,
            "min_features": args.min_features,
            "cap": args.cap,
            "feature_kinds": sorted(feature_kind_set(args.feature_kind or [])),
            "include_sources": sorted(include_sources),
            "exclude_sources": sorted(exclude_sources),
        },
        "outputs": {
            "newick": str(newick_path),
            "figure": None if args.skip_figure else str(png_path),
            "taxa_csv": str(taxa_path),
            "introgression_json": str(introgression_path),
        },
    }
    write_json(summary_path, summary)

    print_introgression_summary(introgression)
    print("\nbranches:", dict(branch_counts))
    print(f"\nwrote {newick_path}")
    if not args.skip_figure:
        print(f"wrote {png_path}")
    print(f"wrote {taxa_path}")
    print(f"wrote {summary_path}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Path to finished RELISH graph. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--out-prefix",
        "--out",
        dest="out_prefix",
        type=Path,
        default=DEFAULT_OUT_PREFIX,
        help=f"Output prefix. Default: {DEFAULT_OUT_PREFIX}",
    )
    parser.add_argument(
        "--keep-all-sefardi",
        action="store_true",
        help="Include all Sephardic recipes instead of only stew-like titles.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected taxa and exit without building a tree.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=4,
        help="Minimum character features required for a recipe to enter the tree.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=45,
        help="Maximum number of taxa. Use 0 for no cap.",
    )
    parser.add_argument(
        "--feature-kind",
        action="append",
        choices=["ingredient", "foodon", "action", "tool"],
        help=(
            "Feature type to include. Can be repeated. "
            "Default: ingredient, foodon, action."
        ),
    )
    parser.add_argument(
        "--include-source",
        action="append",
        help="Restrict lineage selection to this source_id. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        help="Exclude this source_id from lineage selection. Can be repeated.",
    )
    parser.add_argument(
        "--extra-node",
        action="append",
        default=[],
        help="Explicit Recipe node id to include. Can be repeated.",
    )
    parser.add_argument(
        "--extra-node-file",
        type=Path,
        help="Text file with one explicit Recipe node id per line.",
    )
    parser.add_argument(
        "--new-world-markers",
        type=Path,
        help="Optional marker file passed through to recipe_phylogenetics.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write Newick/CSV/JSON but do not draw PNG.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.min_features < 1:
        log.error("--min-features must be >= 1")
        return 2
    if args.cap < 0:
        log.error("--cap must be >= 0")
        return 2

    try:
        return run(args)
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
