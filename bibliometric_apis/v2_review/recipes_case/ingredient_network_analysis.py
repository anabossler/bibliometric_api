"""
==============================

Ingredient Network Analysis 

This script performs a reproducible analysis of the historical recipe
ingredient co-occurrence network.

Analyses
--------

1. Bipartite projection:
   Recipe -> Ingredient edges are projected into an Ingredient-Ingredient
   co-occurrence graph.

2. Centrality metrics:
   degree, weighted degree, betweenness, eigenvector centrality, and PageRank.

3. Community detection:
   Louvain when available; otherwise NetworkX Louvain/greedy modularity/
   connected components as fallbacks.

4. Temporal evolution:
   ingredient usage frequency by derived historical period.

5. FoodOn semantic grouping:
   community composition by mapped FoodOn ancestors.

6. Place-based analysis:
   ingredient profiles by source_place.

Default input
-------------

  data/graph_step4_foodon.gpickle

Default outputs
---------------

  analysis/ingredient_centrality.csv
  analysis/ingredient_communities.csv
  analysis/temporal_ingredient_evolution.csv
  analysis/cooccurrence_top_pairs.csv
  analysis/place_ingredient_profiles.csv
  analysis/foodon_community_composition.csv
  analysis/summary_stats.json

  figs/centrality_top30.png
  figs/temporal_heatmap.png
  figs/cooccurrence_network_top.png

Usage
-----

  python ingredient_network_analysis.py

  python ingredient_network_analysis.py --graph data/graph_step3_flagged.gpickle

  python ingredient_network_analysis.py --min-cooccurrence 5 --top-n 100

  python ingredient_network_analysis.py --skip-figures

Dependencies
------------

  pip install pandas numpy matplotlib networkx

Optional:

  pip install python-louvain


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover - optional dependency
    community_louvain = None


DEFAULT_GRAPH = Path("data/graph_step4_foodon.gpickle")
DEFAULT_OUTPUT_DIR = Path("analysis")
DEFAULT_FIG_DIR = Path("figs")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("ingredient_network_analysis")


PERIOD_ORDER = [
    "ancient",
    "medieval_early",
    "13c",
    "14c",
    "15c",
    "16c",
    "17c",
    "18c",
    "19c",
    "20c",
    "21c",
]

PERIOD_LABELS = {
    "ancient": "Ancient\n(≤500)",
    "medieval_early": "Early Med.\n(501–1199)",
    "13c": "13th c.",
    "14c": "14th c.",
    "15c": "15th c.",
    "16c": "16th c.",
    "17c": "17th c.",
    "18c": "18th c.",
    "19c": "19th c.",
    "20c": "20th c.",
    "21c": "21st c.",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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

    type_counts = Counter(attrs.get("node_type", "?") for _, attrs in graph.nodes(data=True))
    for node_type, count in type_counts.most_common():
        log.info("  %s: %d", node_type, count)

    return graph


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def node_label(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes.get(node_id, {})
    return safe_str(
        attrs.get("canonical_name")
        or attrs.get("canonical_verb")
        or attrs.get("label")
        or attrs.get("name")
        or str(node_id).replace("ing::", "")
    )


def extract_recipe_ingredients(graph: nx.DiGraph) -> dict[str, list[str]]:
    """Return {recipe_id: [ingredient_ids]} from contains edges."""
    recipe_ingredients: dict[str, list[str]] = defaultdict(list)

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "contains":
            continue
        if graph.nodes.get(source, {}).get("node_type") != "Recipe":
            continue
        if graph.nodes.get(target, {}).get("node_type") != "Ingredient":
            continue

        recipe_ingredients[str(source)].append(str(target))

    return dict(recipe_ingredients)


# ---------------------------------------------------------------------------
# Co-occurrence graph
# ---------------------------------------------------------------------------

def build_cooccurrence_graph(
    recipe_ingredients: dict[str, list[str]],
    full_graph: nx.DiGraph,
    *,
    min_cooccurrence: int,
) -> nx.Graph:
    """Project Recipe-Ingredient edges into Ingredient-Ingredient co-occurrence.

    Edge weight = number of recipes where both ingredients appear together.
    A reciprocal distance attribute is also stored for shortest-path metrics.
    """
    if min_cooccurrence < 1:
        raise ValueError("--min-cooccurrence must be >= 1")

    log.info("Building co-occurrence graph with min_cooccurrence=%d", min_cooccurrence)

    pair_counts: Counter[tuple[str, str]] = Counter()

    for ingredients in recipe_ingredients.values():
        unique_ingredients = sorted(set(ingredients))
        for ingredient_a, ingredient_b in combinations(unique_ingredients, 2):
            pair_counts[(ingredient_a, ingredient_b)] += 1

    co_graph = nx.Graph()

    for (ingredient_a, ingredient_b), weight in pair_counts.items():
        if weight < min_cooccurrence:
            continue

        co_graph.add_node(ingredient_a, name=node_label(full_graph, ingredient_a))
        co_graph.add_node(ingredient_b, name=node_label(full_graph, ingredient_b))
        co_graph.add_edge(
            ingredient_a,
            ingredient_b,
            weight=int(weight),
            distance=1.0 / float(weight),
        )

    log.info(
        "Co-occurrence graph: %d nodes, %d edges",
        co_graph.number_of_nodes(),
        co_graph.number_of_edges(),
    )
    return co_graph


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def ingredient_corpus_frequency(full_graph: nx.DiGraph, ingredient_id: str) -> int:
    return sum(
        1
        for source, _, edge_attrs in full_graph.in_edges(ingredient_id, data=True)
        if edge_attrs.get("edge_type") == "contains"
        and full_graph.nodes.get(source, {}).get("node_type") == "Recipe"
    )


def foodon_leaf_label(full_graph: nx.DiGraph, ingredient_id: str) -> str:
    for _, target, edge_attrs in full_graph.out_edges(ingredient_id, data=True):
        if edge_attrs.get("edge_type") == "mapped_to_foodon":
            return node_label(full_graph, str(target))
    return ""


def compute_centrality(co_graph: nx.Graph, full_graph: nx.DiGraph) -> pd.DataFrame:
    """Compute centrality metrics for the ingredient co-occurrence graph."""
    if co_graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "ingredient_id",
            "name",
            "corpus_frequency",
            "degree",
            "degree_centrality",
            "weighted_degree",
            "betweenness",
            "eigenvector",
            "pagerank",
            "foodon_class",
        ])

    log.info("Computing centrality metrics")

    degree_centrality = nx.degree_centrality(co_graph)

    weighted_degree = {
        node: sum(edge_attrs.get("weight", 1) for _, _, edge_attrs in co_graph.edges(node, data=True))
        for node in co_graph.nodes()
    }

    # Important: co-occurrence weight is strength, not distance. Betweenness
    # should therefore use the reciprocal distance attribute.
    betweenness = nx.betweenness_centrality(
        co_graph,
        weight="distance",
        normalized=True,
    )

    try:
        eigenvector = nx.eigenvector_centrality(
            co_graph,
            weight="weight",
            max_iter=1000,
            tol=1e-06,
        )
    except Exception as exc:
        log.warning("Eigenvector centrality failed; using degree centrality fallback: %s", exc)
        eigenvector = degree_centrality

    pagerank = nx.pagerank(co_graph, weight="weight")

    rows: list[dict[str, Any]] = []
    for node in sorted(co_graph.nodes()):
        rows.append({
            "ingredient_id": node,
            "name": co_graph.nodes[node].get("name", node.replace("ing::", "")),
            "corpus_frequency": ingredient_corpus_frequency(full_graph, node),
            "degree": int(co_graph.degree(node)),
            "degree_centrality": round(float(degree_centrality.get(node, 0.0)), 8),
            "weighted_degree": int(weighted_degree.get(node, 0)),
            "betweenness": round(float(betweenness.get(node, 0.0)), 8),
            "eigenvector": round(float(eigenvector.get(node, 0.0)), 8),
            "pagerank": round(float(pagerank.get(node, 0.0)), 8),
            "foodon_class": foodon_leaf_label(full_graph, node),
        })

    df = pd.DataFrame(rows).sort_values(
        by=["eigenvector", "pagerank", "weighted_degree", "name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )

    if not df.empty:
        log.info("Top central ingredients: %s", ", ".join(df.head(10)["name"].tolist()))

    return df


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(co_graph: nx.Graph, *, seed: int) -> tuple[dict[str, int], dict[str, Any]]:
    """Detect ingredient communities with deterministic fallbacks."""
    if co_graph.number_of_nodes() == 0:
        return {}, {"method": "none", "n_communities": 0, "modularity": None}

    if community_louvain is not None:
        log.info("Running Louvain community detection with python-louvain")
        partition = community_louvain.best_partition(co_graph, weight="weight", random_state=seed)
        modularity = community_louvain.modularity(partition, co_graph, weight="weight")
        return {
            str(node): int(comm_id)
            for node, comm_id in partition.items()
        }, {
            "method": "python_louvain",
            "n_communities": len(set(partition.values())),
            "modularity": round(float(modularity), 6),
        }

    try:
        from networkx.algorithms.community import louvain_communities

        log.info("Running Louvain community detection with NetworkX")
        communities = louvain_communities(co_graph, weight="weight", seed=seed)
        partition = {
            str(node): int(comm_id)
            for comm_id, community_nodes in enumerate(communities)
            for node in community_nodes
        }
        modularity = nx.algorithms.community.modularity(co_graph, communities, weight="weight")
        return partition, {
            "method": "networkx_louvain",
            "n_communities": len(communities),
            "modularity": round(float(modularity), 6),
        }
    except Exception as exc:
        log.warning("NetworkX Louvain unavailable/failed: %s", exc)

    try:
        log.info("Running greedy modularity community detection")
        communities = list(nx.algorithms.community.greedy_modularity_communities(co_graph, weight="weight"))
        partition = {
            str(node): int(comm_id)
            for comm_id, community_nodes in enumerate(communities)
            for node in community_nodes
        }
        modularity = nx.algorithms.community.modularity(co_graph, communities, weight="weight")
        return partition, {
            "method": "greedy_modularity",
            "n_communities": len(communities),
            "modularity": round(float(modularity), 6),
        }
    except Exception as exc:
        log.warning("Greedy modularity failed: %s", exc)

    log.warning("Using connected components as community fallback")
    partition = {}
    for comm_id, component in enumerate(nx.connected_components(co_graph)):
        for node in component:
            partition[str(node)] = int(comm_id)

    return partition, {
        "method": "connected_components",
        "n_communities": len(set(partition.values())),
        "modularity": None,
    }


# ---------------------------------------------------------------------------
# Temporal and place analyses
# ---------------------------------------------------------------------------

def temporal_evolution(
    full_graph: nx.DiGraph,
    recipe_ingredients: dict[str, list[str]],
    *,
    top_n: int,
) -> pd.DataFrame:
    """Track ingredient usage frequency across periods."""
    if top_n < 1:
        raise ValueError("--top-n must be >= 1")

    log.info("Computing temporal evolution for top %d ingredients", top_n)

    recipe_period: dict[str, str] = {}
    for node, attrs in full_graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue
        period = attrs.get("period_derived")
        if period:
            recipe_period[str(node)] = str(period)

    period_recipe_counts = Counter(recipe_period.values())
    log.info("Recipes by period: %s", {p: period_recipe_counts.get(p, 0) for p in PERIOD_ORDER})

    ingredient_period_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for recipe_id, ingredients in recipe_ingredients.items():
        period = recipe_period.get(recipe_id)
        if not period:
            continue
        for ingredient in set(ingredients):
            ingredient_period_counts[ingredient][period] += 1

    total_frequency = Counter({
        ingredient: sum(period_counts.values())
        for ingredient, period_counts in ingredient_period_counts.items()
    })

    top_ingredients = [ingredient for ingredient, _ in total_frequency.most_common(top_n)]

    rows: list[dict[str, Any]] = []
    for ingredient in top_ingredients:
        row: dict[str, Any] = {
            "ingredient": node_label(full_graph, ingredient),
            "ingredient_id": ingredient,
            "total_count": int(total_frequency[ingredient]),
        }

        for period in PERIOD_ORDER:
            raw_count = int(ingredient_period_counts[ingredient].get(period, 0))
            n_recipes = int(period_recipe_counts.get(period, 0))
            pct = round(100 * raw_count / n_recipes, 4) if n_recipes else 0.0

            row[f"{period}_count"] = raw_count
            row[f"{period}_pct"] = pct

        rows.append(row)

    return pd.DataFrame(rows)


def place_profiles(
    full_graph: nx.DiGraph,
    recipe_ingredients: dict[str, list[str]],
    *,
    top_n_ingredients: int,
    min_recipes_per_place: int,
) -> pd.DataFrame:
    """Compute ingredient profiles by source_place."""
    if top_n_ingredients < 1:
        raise ValueError("--place-top-ingredients must be >= 1")
    if min_recipes_per_place < 1:
        raise ValueError("--place-min-recipes must be >= 1")

    log.info("Computing place-based ingredient profiles")

    recipe_place: dict[str, str] = {}
    for node, attrs in full_graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue
        place = safe_str(attrs.get("source_place"))
        if place:
            recipe_place[str(node)] = place

    place_recipe_counts = Counter(recipe_place.values())
    valid_places = {
        place
        for place, count in place_recipe_counts.items()
        if count >= min_recipes_per_place
    }

    log.info("Places with >= %d recipes: %d", min_recipes_per_place, len(valid_places))

    place_ingredient_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for recipe_id, ingredients in recipe_ingredients.items():
        place = recipe_place.get(recipe_id)
        if not place or place not in valid_places:
            continue
        for ingredient in set(ingredients):
            place_ingredient_counts[place][ingredient] += 1

    global_frequency = Counter()
    for counts in place_ingredient_counts.values():
        global_frequency.update(counts)

    top_ingredients = [ingredient for ingredient, _ in global_frequency.most_common(top_n_ingredients)]

    rows: list[dict[str, Any]] = []
    for place in sorted(valid_places):
        row: dict[str, Any] = {
            "place": place,
            "n_recipes": int(place_recipe_counts[place]),
        }

        for ingredient in top_ingredients:
            label = node_label(full_graph, ingredient)
            count = int(place_ingredient_counts[place].get(ingredient, 0))
            row[label] = round(100 * count / place_recipe_counts[place], 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Co-occurrence pairs and FoodOn composition
# ---------------------------------------------------------------------------

def top_cooccurrence_pairs(co_graph: nx.Graph, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ingredient_a, ingredient_b, edge_attrs in co_graph.edges(data=True):
        rows.append({
            "ingredient_a_id": ingredient_a,
            "ingredient_b_id": ingredient_b,
            "ingredient_a": co_graph.nodes[ingredient_a].get("name", ingredient_a.replace("ing::", "")),
            "ingredient_b": co_graph.nodes[ingredient_b].get("name", ingredient_b.replace("ing::", "")),
            "cooccurrence_count": int(edge_attrs.get("weight", 0)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ingredient_a_id",
            "ingredient_b_id",
            "ingredient_a",
            "ingredient_b",
            "cooccurrence_count",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["cooccurrence_count", "ingredient_a", "ingredient_b"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def foodon_ancestors(full_graph: nx.DiGraph, foodon_node: str) -> set[str]:
    """Return FoodOn leaf + transitive is_a ancestors."""
    seen: set[str] = set()
    stack = [foodon_node]

    while stack:
        current = stack.pop()

        if current in seen:
            continue
        seen.add(current)

        for _, parent, edge_attrs in full_graph.out_edges(current, data=True):
            if edge_attrs.get("edge_type") == "is_a":
                stack.append(str(parent))

    return seen


def foodon_community_analysis(
    full_graph: nx.DiGraph,
    partition: dict[str, int],
) -> pd.DataFrame:
    """For each ingredient community, count mapped FoodOn ancestor labels."""
    log.info("Analysing FoodOn composition per community")

    ingredient_to_ancestor_labels: dict[str, set[str]] = defaultdict(set)

    for source, target, edge_attrs in full_graph.edges(data=True):
        if edge_attrs.get("edge_type") != "mapped_to_foodon":
            continue
        if full_graph.nodes.get(source, {}).get("node_type") != "Ingredient":
            continue
        if full_graph.nodes.get(target, {}).get("node_type") != "FoodOnClass":
            continue

        for ancestor in foodon_ancestors(full_graph, str(target)):
            label = node_label(full_graph, ancestor)
            if label:
                ingredient_to_ancestor_labels[str(source)].add(label)

    community_foodon: dict[int, Counter[str]] = defaultdict(Counter)
    community_members: Counter[int] = Counter()
    community_mapped_members: Counter[int] = Counter()

    for ingredient_id, community_id in partition.items():
        community_members[int(community_id)] += 1

        labels = ingredient_to_ancestor_labels.get(ingredient_id, set())
        if labels:
            community_mapped_members[int(community_id)] += 1

        for label in labels:
            community_foodon[int(community_id)][label] += 1

    rows: list[dict[str, Any]] = []
    for community_id in sorted(community_members):
        top_categories = community_foodon[community_id].most_common(5)

        rows.append({
            "community_id": int(community_id),
            "n_ingredients": int(community_members[community_id]),
            "n_with_foodon": int(community_mapped_members[community_id]),
            "top_foodon_categories": "; ".join(
                f"{category} ({count})"
                for category, count in top_categories
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_centrality_plot(centrality_df: pd.DataFrame, fig_dir: Path, *, top_n: int) -> None:
    if centrality_df.empty:
        log.warning("Skipping centrality plot: centrality table is empty")
        return

    top = centrality_df.head(top_n).sort_values("eigenvector", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(top))))
    ax.barh(top["name"], top["eigenvector"])
    ax.set_xlabel("Eigenvector centrality")
    ax.set_title(f"Top {len(top)} ingredients by eigenvector centrality")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    path = fig_dir / "centrality_top30.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def save_temporal_heatmap(temporal_df: pd.DataFrame, fig_dir: Path) -> None:
    if temporal_df.empty:
        log.warning("Skipping temporal heatmap: temporal table is empty")
        return

    pct_columns = [f"{period}_pct" for period in PERIOD_ORDER if f"{period}_pct" in temporal_df.columns]
    if not pct_columns:
        log.warning("Skipping temporal heatmap: no *_pct columns")
        return

    matrix = temporal_df.set_index("ingredient")[pct_columns].to_numpy(dtype=float)
    labels_y = temporal_df["ingredient"].tolist()
    labels_x = [PERIOD_LABELS.get(col.replace("_pct", ""), col.replace("_pct", "")) for col in pct_columns]

    fig, ax = plt.subplots(figsize=(12, max(8, 0.32 * len(labels_y))))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=7)
    ax.set_title("Ingredient usage across historical periods")
    fig.colorbar(image, ax=ax, label="% of recipes in period")
    fig.tight_layout()

    path = fig_dir / "temporal_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def save_cooccurrence_network_plot(
    co_graph: nx.Graph,
    partition: dict[str, int],
    fig_dir: Path,
    *,
    top_n: int,
    seed: int,
) -> None:
    if co_graph.number_of_nodes() == 0:
        log.warning("Skipping co-occurrence network plot: graph is empty")
        return

    node_strength = {
        node: sum(edge_attrs.get("weight", 1) for _, _, edge_attrs in co_graph.edges(node, data=True))
        for node in co_graph.nodes()
    }

    top_nodes = sorted(node_strength, key=lambda node: (-node_strength[node], str(node)))[:top_n]
    subgraph = co_graph.subgraph(top_nodes).copy()

    if subgraph.number_of_nodes() == 0:
        log.warning("Skipping co-occurrence network plot: selected subgraph is empty")
        return

    pos = nx.spring_layout(subgraph, weight="weight", k=1.5, iterations=80, seed=seed)

    community_values = [partition.get(str(node), 0) for node in subgraph.nodes()]
    max_strength = max(node_strength[node] for node in subgraph.nodes()) if subgraph.nodes else 1
    node_sizes = [120 + 600 * node_strength[node] / max_strength for node in subgraph.nodes()]

    edge_weights = [edge_attrs.get("weight", 1) for _, _, edge_attrs in subgraph.edges(data=True)]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.2 + 2.0 * weight / max_edge_weight for weight in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.25, width=edge_widths)
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_color=community_values,
        node_size=node_sizes,
        linewidths=0.5,
    )
    labels = {
        node: subgraph.nodes[node].get("name", str(node).replace("ing::", ""))
        for node in subgraph.nodes()
    }
    nx.draw_networkx_labels(subgraph, pos, labels, ax=ax, font_size=6)
    ax.set_title(f"Ingredient co-occurrence network, top {len(top_nodes)} ingredients")
    ax.axis("off")
    fig.tight_layout()

    path = fig_dir / "cooccurrence_network_top.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Path to enriched graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--min-cooccurrence",
        type=int,
        default=3,
        help="Minimum co-occurrence count for ingredient-ingredient edges.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top ingredients for temporal analysis and centrality plot.",
    )
    parser.add_argument(
        "--cooccurrence-top-pairs",
        type=int,
        default=200,
        help="Number of top co-occurrence pairs to export.",
    )
    parser.add_argument(
        "--place-top-ingredients",
        type=int,
        default=20,
        help="Number of global top ingredients used in place profiles.",
    )
    parser.add_argument(
        "--place-min-recipes",
        type=int,
        default=10,
        help="Minimum recipes required for a source_place to appear in place profiles.",
    )
    parser.add_argument(
        "--community-network-top",
        type=int,
        default=80,
        help="Number of ingredients shown in the co-occurrence network figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/JSON outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Directory for figure outputs. Default: {DEFAULT_FIG_DIR}",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Run analyses but do not generate PNG figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for layouts/community algorithms where applicable.",
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

    if args.min_cooccurrence < 1:
        log.error("--min-cooccurrence must be >= 1")
        return 2
    if args.top_n < 1:
        log.error("--top-n must be >= 1")
        return 2
    if args.cooccurrence_top_pairs < 1:
        log.error("--cooccurrence-top-pairs must be >= 1")
        return 2
    if args.place_top_ingredients < 1:
        log.error("--place-top-ingredients must be >= 1")
        return 2
    if args.place_min_recipes < 1:
        log.error("--place-min-recipes must be >= 1")
        return 2
    if args.community_network_top < 1:
        log.error("--community-network-top must be >= 1")
        return 2

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.fig_dir.mkdir(parents=True, exist_ok=True)

        full_graph = load_graph(args.graph)
        recipe_ingredients = extract_recipe_ingredients(full_graph)
        log.info("Extracted ingredient lists for %d recipes", len(recipe_ingredients))

        co_graph = build_cooccurrence_graph(
            recipe_ingredients,
            full_graph,
            min_cooccurrence=args.min_cooccurrence,
        )

        if co_graph.number_of_nodes() == 0:
            raise ValueError(
                "Co-occurrence graph is empty. Lower --min-cooccurrence or check contains edges."
            )

        centrality_df = compute_centrality(co_graph, full_graph)
        partition, community_info = detect_communities(co_graph, seed=args.seed)

        centrality_with_communities = centrality_df.copy()
        centrality_with_communities["community"] = centrality_with_communities["ingredient_id"].map(partition)

        temporal_df = temporal_evolution(
            full_graph,
            recipe_ingredients,
            top_n=args.top_n,
        )

        pairs_df = top_cooccurrence_pairs(
            co_graph,
            top_n=args.cooccurrence_top_pairs,
        )

        place_df = place_profiles(
            full_graph,
            recipe_ingredients,
            top_n_ingredients=args.place_top_ingredients,
            min_recipes_per_place=args.place_min_recipes,
        )

        foodon_df = foodon_community_analysis(full_graph, partition)

        write_csv(centrality_df, args.output_dir / "ingredient_centrality.csv")
        write_csv(centrality_with_communities, args.output_dir / "ingredient_communities.csv")
        write_csv(temporal_df, args.output_dir / "temporal_ingredient_evolution.csv")
        write_csv(pairs_df, args.output_dir / "cooccurrence_top_pairs.csv")
        write_csv(place_df, args.output_dir / "place_ingredient_profiles.csv")
        write_csv(foodon_df, args.output_dir / "foodon_community_composition.csv")

        try:
            avg_clustering = nx.average_clustering(co_graph, weight="weight")
        except ZeroDivisionError:
            avg_clustering = 0.0

        summary = {
            "input_graph": str(args.graph),
            "n_recipes": len(recipe_ingredients),
            "n_ingredients_total": sum(
                1
                for _, attrs in full_graph.nodes(data=True)
                if attrs.get("node_type") == "Ingredient"
            ),
            "n_ingredients_in_cooccurrence": co_graph.number_of_nodes(),
            "n_cooccurrence_edges": co_graph.number_of_edges(),
            "min_cooccurrence_threshold": args.min_cooccurrence,
            "community_detection": community_info,
            "graph_density": round(float(nx.density(co_graph)), 8),
            "avg_clustering": round(float(avg_clustering), 8),
            "n_connected_components": nx.number_connected_components(co_graph),
            "top_10_central": centrality_df.head(10)[
                ["name", "eigenvector", "pagerank", "weighted_degree"]
            ].to_dict("records"),
        }

        write_json(args.output_dir / "summary_stats.json", summary)

        if not args.skip_figures:
            log.info("Generating figures")
            save_centrality_plot(centrality_df, args.fig_dir, top_n=args.top_n)
            save_temporal_heatmap(temporal_df, args.fig_dir)
            save_cooccurrence_network_plot(
                co_graph,
                partition,
                args.fig_dir,
                top_n=args.community_network_top,
                seed=args.seed,
            )

        log.info("=" * 60)
        log.info("ANALYSIS COMPLETE")
        log.info("=" * 60)
        log.info("Recipes analysed:       %d", summary["n_recipes"])
        log.info("Ingredients in co-occ:  %d", summary["n_ingredients_in_cooccurrence"])
        log.info("Co-occurrence edges:    %d", summary["n_cooccurrence_edges"])
        log.info("Communities detected:   %d", summary["community_detection"]["n_communities"])
        log.info("Output files in:        %s", args.output_dir)
        if not args.skip_figures:
            log.info("Figures in:             %s", args.fig_dir)

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        pickle.PickleError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
