"""
=======================

Exploratory recipe "phylogenetics" for the RELISH graph.

This module treats recipes as taxa and recipe features as binary characters.
It builds a presence/absence character matrix from ingredients, FoodOn classes,
actions, and tools; computes pairwise Jaccard distances; and reconstructs an
exploratory Neighbor-Joining tree.

Conceptual mapping
------------------

  biology                     ->  cuisine / recipe history
  ----------------------------    ------------------------------------------
  organism / taxon            ->  one recipe
  genome / character matrix   ->  presence/absence of ingredients, FoodOn
                                  classes, techniques, and tools
  molecular character         ->  one ingredient/technique/tool/ontology column
  point mutation              ->  ingredient substitution
  selective pressure          ->  religious law, climate, migration,
                                  availability, market access
  horizontal gene transfer    ->  New World ingredients entering multiple
                                  lineages after 1492
  outgroup / root             ->  oldest recipe, used only as time polarization

Important caution
-----------------

This is not biological phylogenetics and should not be interpreted as proof of
descent. It is a structured exploratory analogy: the tree summarizes similarity
among recipe character profiles. It is useful for hypothesis generation,
visualization, and detecting ingredient-introduction patterns.

Default input
-------------

  data/graph_step5_audited.gpickle

Typical usage
-------------

  python recipe_phylogenetics.py \
    --graph data/graph_step5_audited.gpickle \
    --out-prefix analysis/phylo_demo

  python recipe_phylogenetics.py \
    --graph data/graph_step5_audited.gpickle \
    --title-regex "dafina|cocido|olla|potaje" \
    --cap 45 \
    --out-prefix analysis/dafina_cocido_phylo

Outputs
-------

  <out-prefix>.nwk
  <out-prefix>.png                     unless --skip-figure
  <out-prefix>_taxa.csv
  <out-prefix>_characters.csv
  <out-prefix>_distances.csv
  <out-prefix>_introgression.json
  <out-prefix>_summary.json

Dependencies
------------

  pip install numpy pandas networkx biopython matplotlib


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step5_audited.gpickle")
DEFAULT_OUT_PREFIX = Path("analysis/phylo_demo")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("recipe_phylogenetics")

DEFAULT_NEW_WORLD = {
    "potato", "tomato", "maize", "corn", "capsicum", "chili", "chile",
    "chilli", "bell pepper", "sweet pepper", "paprika", "cacao",
    "chocolate", "vanilla", "common bean", "kidney bean", "turkey",
    "pumpkin", "squash", "peanut", "cassava", "pineapple", "avocado",
}

PERIOD_ORDER = [
    "ancient", "medieval_early", "13c", "14c", "15c", "16c", "17c",
    "18c", "19c", "20c", "21c",
]

_YEAR_RE = re.compile(r"-?\d{1,4}")


def load_graph(path: Path) -> nx.DiGraph:
    """Load a trusted local NetworkX graph.

    Do not use this function with untrusted pickle/gpickle files.
    """
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def parse_year(value: Any) -> int | None:
    """Parse common year formats: 1520, 'c. 1520', '1520-1530'."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    match = _YEAR_RE.search(str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_token(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def node_label(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes.get(node_id, {})
    return normalize_token(
        attrs.get("canonical_name")
        or attrs.get("canonical_verb")
        or attrs.get("label")
        or attrs.get("name")
        or str(node_id).replace("ing::", "").replace("act::", "").replace("tool::", "")
    )


def recipe_meta(graph: nx.Graph, recipe_id: str) -> dict[str, Any]:
    attrs = graph.nodes[recipe_id]
    raw_year = attrs.get("source_year") if attrs.get("source_year") is not None else attrs.get("year")
    return {
        "id": recipe_id,
        "title": attrs.get("title") or "",
        "source_id": attrs.get("source_id"),
        "source_title": attrs.get("source_title"),
        "source_author": attrs.get("source_author"),
        "year": parse_year(raw_year),
        "raw_year": raw_year,
        "period": attrs.get("period_derived"),
        "place": attrs.get("source_place"),
        "lang": attrs.get("source_language"),
    }


def make_tip_label(index: int, meta: dict[str, Any]) -> str:
    """Create a readable, unique, Newick-friendly tip label."""
    year = meta["year"] if meta["year"] is not None else "unknown_year"
    title = safe_str(meta.get("title"), "untitled")
    title = re.sub(r"[^A-Za-z0-9]+", "_", title[:32]).strip("_") or "untitled"
    source = re.sub(r"[^A-Za-z0-9]+", "_", safe_str(meta.get("source_id"), "source"))[:20]
    return f"t{index:03d}_{title}_{source}_{year}"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def foodon_ancestors(graph: nx.Graph, foodon_node: str, *, include_leaf: bool = True) -> set[str]:
    """Return FoodOn leaf and transitive is_a ancestors.

    Expected graph pattern:
      Ingredient --mapped_to_foodon--> FoodOnClass
      FoodOnClass --is_a--> FoodOnClass parent
    """
    result: set[str] = set()
    seen: set[str] = set()
    stack = [foodon_node]

    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if include_leaf or current != foodon_node:
            result.add(str(current))
        for _, parent, edge_attrs in graph.out_edges(current, data=True):
            if edge_attrs.get("edge_type") == "is_a":
                if graph.nodes.get(parent, {}).get("node_type") == "FoodOnClass":
                    stack.append(str(parent))
    return result


def ingredient_foodon_features(graph: nx.Graph, ingredient_id: str) -> set[str]:
    """Return FoodOn feature tokens for one Ingredient node."""
    features: set[str] = set()
    for _, foodon_node, edge_attrs in graph.out_edges(ingredient_id, data=True):
        if edge_attrs.get("edge_type") != "mapped_to_foodon":
            continue
        if graph.nodes.get(foodon_node, {}).get("node_type") != "FoodOnClass":
            continue
        for cls in foodon_ancestors(graph, str(foodon_node), include_leaf=True):
            label = node_label(graph, cls)
            if label:
                features.add(f"fdn::{label}")
    return features


def recipe_features(graph: nx.Graph, recipe_id: str, *, feature_kinds: set[str]) -> set[str]:
    """Return namespaced character tokens for one recipe.

    Namespaces:
      ing::   ingredient
      fdn::   FoodOn class / ancestor
      act::   action / technique
      tool::  tool
    """
    features: set[str] = set()
    if not graph.has_node(recipe_id):
        return features

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        target = str(target)
        target_attrs = graph.nodes.get(target, {})
        node_type = target_attrs.get("node_type")
        edge_type = edge_attrs.get("edge_type")

        if node_type == "Ingredient" and edge_type == "contains":
            if "ingredient" in feature_kinds:
                label = node_label(graph, target)
                if label:
                    features.add(f"ing::{label}")
            if "foodon" in feature_kinds:
                features.update(ingredient_foodon_features(graph, target))

        elif node_type == "Action" and edge_type == "performs" and "action" in feature_kinds:
            label = node_label(graph, target)
            if label:
                features.add(f"act::{label}")

        elif node_type == "Tool" and edge_type == "uses_tool" and "tool" in feature_kinds:
            label = node_label(graph, target)
            if label:
                features.add(f"tool::{label}")

    return features


# ---------------------------------------------------------------------------
# Selection and matrices
# ---------------------------------------------------------------------------

def load_node_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Node list not found: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def select_taxa(
    graph: nx.Graph,
    *,
    title_regex: str | None = None,
    source_ids: set[str] | None = None,
    place_keywords: set[str] | None = None,
    extra_nodes: set[str] | None = None,
    min_features: int = 3,
    cap: int | None = None,
    feature_kinds: set[str] | None = None,
) -> list[tuple[str, dict[str, Any], set[str]]]:
    """Return [(recipe_id, metadata, feature_set), ...] for selected taxa."""
    if min_features < 1:
        raise ValueError("--min-features must be >= 1")

    feature_kinds = feature_kinds or {"ingredient", "foodon", "action", "tool"}
    source_ids = source_ids or set()
    place_keywords = {x.lower() for x in (place_keywords or set())}
    extra_nodes = extra_nodes or set()
    pattern = re.compile(title_regex, re.IGNORECASE) if title_regex else None

    selected: list[tuple[str, dict[str, Any], set[str]]] = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        recipe_id = str(node_id)
        title = safe_str(attrs.get("title"))
        source_id = safe_str(attrs.get("source_id"))
        place = safe_str(attrs.get("source_place")).lower()

        keep = False
        if not any([pattern, source_ids, place_keywords, extra_nodes]):
            keep = True
        if recipe_id in extra_nodes:
            keep = True
        if pattern and pattern.search(title):
            keep = True
        if source_ids and source_id in source_ids:
            keep = True
        if place_keywords and any(keyword in place for keyword in place_keywords):
            keep = True
        if not keep:
            continue

        features = recipe_features(graph, recipe_id, feature_kinds=feature_kinds)
        if len(features) < min_features:
            continue
        selected.append((recipe_id, recipe_meta(graph, recipe_id), features))

    # Deduplicate exact same title + feature set, common in reprints.
    seen: set[tuple[str, frozenset[str]]] = set()
    unique: list[tuple[str, dict[str, Any], set[str]]] = []
    for recipe_id, meta, features in selected:
        key = (safe_str(meta.get("title")).lower(), frozenset(features))
        if key in seen:
            continue
        seen.add(key)
        unique.append((recipe_id, meta, features))

    unique.sort(
        key=lambda row: (
            row[1]["year"] if row[1]["year"] is not None else 999999,
            safe_str(row[1].get("source_id")),
            safe_str(row[1].get("title")),
            row[0],
        )
    )

    if cap is not None and cap > 0 and len(unique) > cap:
        unique.sort(
            key=lambda row: (
                -len(row[2]),
                row[1]["year"] if row[1]["year"] is not None else 999999,
                safe_str(row[1].get("title")),
                row[0],
            )
        )
        unique = unique[:cap]
        unique.sort(
            key=lambda row: (
                row[1]["year"] if row[1]["year"] is not None else 999999,
                safe_str(row[1].get("source_id")),
                safe_str(row[1].get("title")),
                row[0],
            )
        )

    return unique


def build_character_matrix(
    taxa: list[tuple[str, dict[str, Any], set[str]]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Return labels, character names, and binary matrix [n_taxa x n_chars]."""
    characters = sorted({feature for _, _, features in taxa for feature in features})
    char_index = {char: index for index, char in enumerate(characters)}
    matrix = np.zeros((len(taxa), len(characters)), dtype=np.uint8)
    labels: list[str] = []

    for row_index, (_, meta, features) in enumerate(taxa):
        labels.append(make_tip_label(row_index, meta))
        for feature in features:
            matrix[row_index, char_index[feature]] = 1
    return labels, characters, matrix


def character_matrix_dataframe(labels: list[str], characters: list[str], matrix: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(matrix, columns=characters)
    df.insert(0, "taxon_label", labels)
    return df


def distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard distance. 0 = identical, 1 = no shared characters."""
    n_taxa = matrix.shape[0]
    distances = np.zeros((n_taxa, n_taxa), dtype=float)
    binary = matrix.astype(bool)

    for i in range(n_taxa):
        a = binary[i]
        for j in range(i + 1, n_taxa):
            b = binary[j]
            intersection = np.count_nonzero(a & b)
            union = np.count_nonzero(a | b)
            distance = 1.0 - (intersection / union if union else 0.0)
            distances[i, j] = distances[j, i] = distance
    return distances


def distance_dataframe(labels: list[str], distances: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(distances, index=labels, columns=labels)
    df.insert(0, "taxon_label", labels)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tree construction and export
# ---------------------------------------------------------------------------

def build_tree(labels: list[str], distances: np.ndarray, *, taxa=None):
    """Build a Neighbor-Joining tree with Biopython and root on oldest taxon."""
    try:
        from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    except ImportError as exc:
        raise RuntimeError("Biopython is required for tree construction: pip install biopython") from exc

    lower_triangular = [[float(distances[i, j]) for j in range(i + 1)] for i in range(len(labels))]
    dm = DistanceMatrix(names=list(labels), matrix=lower_triangular)
    tree = DistanceTreeConstructor().nj(dm)

    if taxa is not None:
        years = [
            (label, meta["year"])
            for label, (_, meta, _) in zip(labels, taxa)
            if meta["year"] is not None
        ]
        if years:
            oldest_label = min(years, key=lambda item: item[1])[0]
            try:
                tree.root_with_outgroup(oldest_label)
            except Exception as exc:
                log.warning("Could not root tree with oldest taxon %s: %s", oldest_label, exc)
    tree.ladderize()
    return tree


def export_newick(tree: Any, path: Path) -> None:
    try:
        from Bio import Phylo
    except ImportError as exc:
        raise RuntimeError("Biopython is required for Newick export: pip install biopython") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Phylo.write(tree, str(path), "newick")
    log.info("Wrote Newick tree: %s", path)


# ---------------------------------------------------------------------------
# New World markers
# ---------------------------------------------------------------------------

def load_new_world_markers(path: Path | None) -> set[str]:
    if path is None:
        return {normalize_token(x) for x in DEFAULT_NEW_WORLD}
    if not path.exists():
        raise FileNotFoundError(f"New World marker file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("New World marker JSON must be a list of strings")
        return {normalize_token(str(x)) for x in data if str(x).strip()}
    return {
        normalize_token(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def feature_matches_marker(feature_name: str, markers: set[str]) -> bool:
    token = normalize_token(feature_name)
    return any(marker == token or marker in token or token in marker for marker in markers)


def new_world_hits(features: set[str], markers: set[str]) -> list[str]:
    hits: set[str] = set()
    for feature in features:
        if not feature.startswith("ing::"):
            continue
        ingredient_name = feature.split("::", 1)[1]
        if feature_matches_marker(ingredient_name, markers):
            hits.add(ingredient_name)
    return sorted(hits)


def map_introgression(taxa: list[tuple[str, dict[str, Any], set[str]]], *, markers: set[str]) -> dict[str, Any]:
    """Return New World marker hits by taxon and period."""
    per_taxon: list[dict[str, Any]] = []
    carriers_by_period: Counter[str] = Counter()
    totals_by_period: Counter[str] = Counter()

    for recipe_id, meta, features in taxa:
        hits = new_world_hits(features, markers)
        period = safe_str(meta.get("period"), "unknown")
        totals_by_period[period] += 1
        if hits:
            carriers_by_period[period] += 1
        per_taxon.append({
            "recipe_id": recipe_id,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "period": meta.get("period"),
            "place": meta.get("place"),
            "source_id": meta.get("source_id"),
            "new_world_hits": hits,
            "carries_new_world_marker": bool(hits),
        })

    ordered_periods = PERIOD_ORDER + sorted(set(totals_by_period) - set(PERIOD_ORDER))
    by_period = []
    for period in ordered_periods:
        total = totals_by_period.get(period, 0)
        if not total:
            continue
        carriers = carriers_by_period.get(period, 0)
        by_period.append({
            "period": period,
            "n_taxa": int(total),
            "n_carriers": int(carriers),
            "carrier_pct": round(100 * carriers / total, 2),
        })

    return {"markers": sorted(markers), "per_taxon": per_taxon, "by_period": by_period}


# ---------------------------------------------------------------------------
# Drawing and tables
# ---------------------------------------------------------------------------

def draw_tree(tree: Any, taxa, labels: list[str], out_png: Path, *, title: str, markers: set[str]) -> None:
    """Draw an annotated tree. Starred tips carry New World markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from Bio import Phylo
    except ImportError as exc:
        raise RuntimeError("matplotlib and Biopython are required for drawing") from exc

    meta_by_label: dict[str, tuple[str | None, bool]] = {}
    for label, (_, meta, features) in zip(labels, taxa):
        meta_by_label[label] = (meta.get("period"), bool(new_world_hits(features, markers)))

    period_colors = {
        "ancient": "#7a5195",
        "medieval_early": "#8d6a9f",
        "13c": "#bc5090",
        "14c": "#ef5675",
        "15c": "#ff764a",
        "16c": "#ffa600",
        "17c": "#c9a227",
        "18c": "#6a994e",
        "19c": "#386641",
        "20c": "#1f6f6f",
        "21c": "#1f4e79",
    }

    def label_func(clade):
        return clade.name if clade.is_terminal() else ""

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, max(6, 0.32 * len(taxa))))
    Phylo.draw(tree, axes=ax, do_show=False, label_func=label_func)

    for text in ax.texts:
        raw = text.get_text().strip()
        if raw not in meta_by_label:
            continue
        period, carries = meta_by_label[raw]
        text.set_color(period_colors.get(period, "#333333"))
        text.set_fontsize(8)
        if carries:
            text.set_text("★ " + raw)
            text.set_fontweight("bold")

    ax.set_title(title, fontsize=13)
    try:
        import matplotlib.patches as mpatches
        used_periods = sorted({meta.get("period") for _, meta, _ in taxa if meta.get("period")})
        handles = [mpatches.Patch(color=period_colors.get(p, "#333333"), label=str(p)) for p in used_periods]
        handles.append(mpatches.Patch(color="black", label="★ New World marker"))
        ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Wrote tree figure: %s", out_png)


def taxa_dataframe(labels: list[str], taxa) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, (recipe_id, meta, features) in zip(labels, taxa):
        rows.append({
            "taxon_label": label,
            "recipe_id": recipe_id,
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    *,
    graph_path: Path,
    out_prefix: Path,
    title_regex: str | None,
    source_ids: set[str],
    place_keywords: set[str],
    extra_nodes: set[str],
    min_features: int,
    cap: int | None,
    feature_kinds: set[str],
    new_world_markers: set[str],
    skip_figure: bool,
) -> dict[str, Any]:
    graph = load_graph(graph_path)
    taxa = select_taxa(
        graph,
        title_regex=title_regex,
        source_ids=source_ids,
        place_keywords=place_keywords,
        extra_nodes=extra_nodes,
        min_features=min_features,
        cap=cap,
        feature_kinds=feature_kinds,
    )
    if len(taxa) < 4:
        raise ValueError(
            f"Only {len(taxa)} taxa selected; need at least 4 for a tree. "
            "Loosen selection, lower --min-features, or raise --cap."
        )

    labels, characters, matrix = build_character_matrix(taxa)
    distances = distance_matrix(matrix)
    tree = build_tree(labels, distances, taxa=taxa)

    newick_path = Path(f"{out_prefix}.nwk")
    png_path = Path(f"{out_prefix}.png")
    taxa_path = Path(f"{out_prefix}_taxa.csv")
    characters_path = Path(f"{out_prefix}_characters.csv")
    distances_path = Path(f"{out_prefix}_distances.csv")
    introgression_path = Path(f"{out_prefix}_introgression.json")
    summary_path = Path(f"{out_prefix}_summary.json")

    export_newick(tree, newick_path)
    if not skip_figure:
        draw_tree(
            tree,
            taxa,
            labels,
            png_path,
            title=f"Recipe similarity tree — {len(taxa)} taxa, {len(characters)} characters",
            markers=new_world_markers,
        )

    introgression = map_introgression(taxa, markers=new_world_markers)
    write_csv(taxa_dataframe(labels, taxa), taxa_path)
    write_csv(character_matrix_dataframe(labels, characters, matrix), characters_path)
    write_csv(distance_dataframe(labels, distances), distances_path)
    write_json(introgression_path, introgression)

    summary = {
        "graph": str(graph_path),
        "out_prefix": str(out_prefix),
        "n_taxa": len(taxa),
        "n_characters": len(characters),
        "feature_kinds": sorted(feature_kinds),
        "min_features": min_features,
        "cap": cap,
        "selection": {
            "title_regex": title_regex,
            "source_ids": sorted(source_ids),
            "place_keywords": sorted(place_keywords),
            "n_extra_nodes": len(extra_nodes),
        },
        "outputs": {
            "newick": str(newick_path),
            "figure": None if skip_figure else str(png_path),
            "taxa_csv": str(taxa_path),
            "characters_csv": str(characters_path),
            "distances_csv": str(distances_path),
            "introgression_json": str(introgression_path),
        },
    }
    write_json(summary_path, summary)

    print(f"taxa={len(taxa)}  characters={len(characters)}")
    print("New World marker carriers by period:")
    for row in introgression["by_period"]:
        print(f"  {row['period']:15s} {row['n_carriers']:3d}/{row['n_taxa']:<3d} ({row['carrier_pct']:5.1f}%)")
    print(f"\nwrote {newick_path}")
    if not skip_figure:
        print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help=f"Path to graph gpickle. Default: {DEFAULT_GRAPH}")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX, help=f"Output prefix. Default: {DEFAULT_OUT_PREFIX}")
    parser.add_argument("--title-regex", default=None, help="Regex used to select recipes by title.")
    parser.add_argument("--source-id", action="append", default=[], help="Select recipes from this source_id. Can be repeated.")
    parser.add_argument("--place-kw", action="append", default=[], help="Select recipes whose source_place contains this keyword. Can be repeated.")
    parser.add_argument("--extra-node", action="append", default=[], help="Explicit Recipe node id to include. Can be repeated.")
    parser.add_argument("--extra-node-file", type=Path, help="Text file with one Recipe node id per line.")
    parser.add_argument("--min-features", type=int, default=3, help="Drop recipes with fewer than this many character features.")
    parser.add_argument("--cap", type=int, default=45, help="Maximum selected taxa. Use 0 for no cap.")
    parser.add_argument(
        "--feature-kind",
        action="append",
        choices=["ingredient", "foodon", "action", "tool"],
        help="Feature type to include. Can be repeated. Default: all.",
    )
    parser.add_argument("--new-world-markers", type=Path, help="Optional JSON or text file of New World ingredient markers.")
    parser.add_argument("--skip-figure", action="store_true", help="Export Newick/CSV/JSON but do not draw PNG.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)

    if args.min_features < 1:
        log.error("--min-features must be >= 1")
        return 2
    if args.cap < 0:
        log.error("--cap must be >= 0")
        return 2

    try:
        extra_nodes = set(args.extra_node)
        extra_nodes.update(load_node_list(args.extra_node_file))
        feature_kinds = set(args.feature_kind) if args.feature_kind else {"ingredient", "foodon", "action", "tool"}
        markers = load_new_world_markers(args.new_world_markers)
        run(
            graph_path=args.graph,
            out_prefix=args.out_prefix,
            title_regex=args.title_regex,
            source_ids=set(args.source_id),
            place_keywords=set(args.place_kw),
            extra_nodes=extra_nodes,
            min_features=args.min_features,
            cap=None if args.cap == 0 else args.cap,
            feature_kinds=feature_kinds,
            new_world_markers=markers,
            skip_figure=args.skip_figure,
        )
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, pickle.PickleError, json.JSONDecodeError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
