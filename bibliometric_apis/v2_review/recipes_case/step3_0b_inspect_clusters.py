"""
============================

Inspect recipe clusters semantically for one chosen k_NN value.

For a chosen k, this script re-runs the k-NN + Leiden clustering and writes a
human-readable cluster inspection table.

For each cluster it reports:

  - size
  - language/source/period/place distributions
  - top distinctive ingredients
  - representative recipe titles closest to the cluster centroid

This is the qualitative companion to step3_0_explore_k.py: it helps decide
whether the numerical clustering is culinarily meaningful.

Default inputs
--------------

  data/graph_step2_canonical.gpickle
  data/layer_2_embeddings.npz

Default output
--------------

  data/cluster_inspection_k{K}.csv
  data/cluster_assignments_k{K}.csv
  data/cluster_distinctive_ingredients_k{K}.csv
  data/cluster_inspection_k{K}_summary.json

Usage
-----

  python step3_0b_inspect_clusters.py --k 30

  python step3_0b_inspect_clusters.py --k 50 --top-ingredients 20

  python step3_0b_inspect_clusters.py \
    --graph data/graph_step2_canonical.gpickle \
    --embeddings data/layer_2_embeddings.npz \
    --k 30

Dependencies
------------

  pip install numpy pandas scikit-learn igraph leidenalg


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step2_canonical.gpickle")
DEFAULT_EMBEDDINGS = Path("data/layer_2_embeddings.npz")
DEFAULT_OUTPUT_DIR = Path("data")

DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_INGREDIENTS = 15
DEFAULT_N_REPRESENTATIVES = 3
DEFAULT_MIN_CLUSTER_INGREDIENT_PCT = 0.05

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step3_0b_inspect_clusters")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def load_recipe_embeddings(path: Path, *, allow_pickle: bool) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    log.info("Loading embeddings: %s", path)
    with np.load(path, allow_pickle=allow_pickle) as npz:
        required = {"recipes_node_ids", "recipes_vectors"}
        missing = required - set(npz.files)
        if missing:
            raise ValueError(f"Embeddings NPZ is missing required arrays: {sorted(missing)}")

        ids = npz["recipes_node_ids"].astype(str)
        vectors = np.asarray(npz["recipes_vectors"], dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"recipes_vectors must be 2D, got shape {vectors.shape}")

    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"recipes_node_ids length {len(ids)} does not match "
            f"recipes_vectors rows {vectors.shape[0]}"
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Recipe embeddings contain NaN or Inf values")

    return ids, vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    n_zero = int((norms[:, 0] < 1e-12).sum())
    if n_zero:
        raise ValueError(f"Recipe embeddings contain {n_zero} near-zero vectors")
    return vectors / np.maximum(norms, 1e-12)


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def ingredient_label(graph, ingredient_id: str) -> str:
    attrs = graph.nodes.get(ingredient_id, {})
    return safe_str(
        attrs.get("canonical_name")
        or attrs.get("name")
        or attrs.get("label")
        or str(ingredient_id).replace("ing::", "")
    )


def load_data(
    *,
    graph_path: Path,
    embeddings_path: Path,
    allow_pickle_npz: bool,
) -> tuple[Any, np.ndarray, np.ndarray, pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    graph = load_graph(graph_path)
    all_ids, all_vectors = load_recipe_embeddings(embeddings_path, allow_pickle=allow_pickle_npz)

    in_graph = np.array([str(node_id) in graph for node_id in all_ids], dtype=bool)
    ids = all_ids[in_graph]
    vectors = all_vectors[in_graph]

    if len(ids) == 0:
        raise ValueError("No recipe embeddings match recipe nodes in the graph")

    vectors = normalize_vectors(vectors)

    rows: list[dict[str, Any]] = []
    for node_id in ids:
        attrs = graph.nodes[str(node_id)]
        rows.append({
            "recipe_id": str(node_id),
            "source_id": attrs.get("source_id"),
            "source_language": attrs.get("source_language"),
            "source_place": attrs.get("source_place"),
            "period_derived": attrs.get("period_derived"),
            "source_year": attrs.get("source_year"),
            "title": attrs.get("title"),
        })

    metadata = pd.DataFrame(rows)

    recipe_to_ingredients: dict[str, list[str]] = defaultdict(list)
    ingredient_labels: dict[str, str] = {}

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "contains":
            continue
        if graph.nodes.get(source, {}).get("node_type") != "Recipe":
            continue
        if graph.nodes.get(target, {}).get("node_type") != "Ingredient":
            continue

        recipe_id = str(source)
        ingredient_id = str(target)
        recipe_to_ingredients[recipe_id].append(ingredient_id)
        ingredient_labels[ingredient_id] = ingredient_label(graph, ingredient_id)

    # Remove duplicate ingredients per recipe for document-frequency style scoring.
    recipe_to_ingredients = {
        recipe_id: sorted(set(ingredients))
        for recipe_id, ingredients in recipe_to_ingredients.items()
    }

    log.info("Recipes with embeddings in graph: %d", len(ids))
    log.info("Ingredient lists built for %d recipes", len(recipe_to_ingredients))

    return graph, vectors, ids, metadata, recipe_to_ingredients, ingredient_labels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def build_knn_graph(vectors: np.ndarray, *, k: int):
    """Build weighted undirected igraph k-NN graph.

    Mutual k-NN edges are collapsed by max similarity so weights remain in
    [0, 1], matching step3_0_explore_k.py.
    """
    try:
        import igraph as ig
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "Requires scikit-learn and igraph: pip install scikit-learn igraph"
        ) from exc

    n = vectors.shape[0]
    if n < 2:
        raise ValueError("Need at least two recipes to cluster")

    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"Invalid k={k} for n={n}")

    if k_eff != k:
        log.warning("k=%d is too large for n=%d; using k=%d", k, n, k_eff)

    log.info("Building k-NN graph with k=%d", k_eff)
    start = time.time()

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    edge_weight_by_pair: dict[tuple[int, int], float] = {}

    for i in range(n):
        for rank in range(1, k_eff + 1):
            j = int(indices[i, rank])
            if i == j:
                continue

            similarity = 1.0 - float(distances[i, rank])
            if similarity <= 0:
                continue

            a, b = (i, j) if i < j else (j, i)
            previous = edge_weight_by_pair.get((a, b))
            if previous is None or similarity > previous:
                edge_weight_by_pair[(a, b)] = similarity

    edges = list(edge_weight_by_pair.keys())
    weights = [edge_weight_by_pair[edge] for edge in edges]

    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = weights

    log.info(
        "k-NN graph built in %.1fs: %d nodes, %d edges",
        time.time() - start,
        graph.vcount(),
        graph.ecount(),
    )
    return graph, k_eff


def run_leiden(
    graph,
    *,
    random_state: int,
    resolution: float | None,
) -> tuple[np.ndarray, float]:
    try:
        import leidenalg
    except ImportError as exc:
        raise RuntimeError("Requires leidenalg: pip install leidenalg") from exc

    log.info("Running Leiden")
    start = time.time()

    if resolution is None:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=random_state,
        )
    else:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )

    labels = np.asarray(partition.membership, dtype=int)
    modularity = float(partition.modularity)

    log.info(
        "Leiden completed in %.1fs: %d clusters, Q=%.4f",
        time.time() - start,
        len(set(labels)),
        modularity,
    )

    return labels, modularity


# ---------------------------------------------------------------------------
# Cluster descriptions
# ---------------------------------------------------------------------------

def top_counts(series: pd.Series, n: int) -> dict[str, int]:
    counts = series.dropna().astype(str).value_counts().head(n)
    return {str(index): int(value) for index, value in counts.items()}


def top_counts_str(series: pd.Series, n: int) -> str:
    counts = top_counts(series, n)
    return "; ".join(f"{key}={value}" for key, value in counts.items())


def distinctive_ingredients(
    *,
    cluster_recipe_ids: list[str],
    other_recipe_ids: list[str],
    recipe_to_ingredients: dict[str, list[str]],
    ingredient_labels: dict[str, str],
    top_n: int,
    min_cluster_pct: float,
    smoothing: float,
) -> list[dict[str, Any]]:
    """Score distinctive ingredients using a smoothed prevalence ratio.

    Score = P(ingredient | cluster) / (P(ingredient | other clusters) + smoothing)

    Ingredients must appear in at least min_cluster_pct of cluster recipes.
    """
    cluster_counts: Counter[str] = Counter()
    other_counts: Counter[str] = Counter()

    for recipe_id in cluster_recipe_ids:
        for ingredient_id in set(recipe_to_ingredients.get(recipe_id, [])):
            cluster_counts[ingredient_id] += 1

    for recipe_id in other_recipe_ids:
        for ingredient_id in set(recipe_to_ingredients.get(recipe_id, [])):
            other_counts[ingredient_id] += 1

    n_cluster = max(1, len(cluster_recipe_ids))
    n_other = max(1, len(other_recipe_ids))

    rows: list[dict[str, Any]] = []

    for ingredient_id, cluster_count in cluster_counts.items():
        cluster_pct = cluster_count / n_cluster
        if cluster_pct < min_cluster_pct:
            continue

        other_count = other_counts.get(ingredient_id, 0)
        other_pct = other_count / n_other
        ratio = cluster_pct / (other_pct + smoothing)

        rows.append({
            "ingredient_id": ingredient_id,
            "ingredient": ingredient_labels.get(ingredient_id, ingredient_id.replace("ing::", "")),
            "score": float(ratio),
            "cluster_count": int(cluster_count),
            "other_count": int(other_count),
            "cluster_pct": float(cluster_pct),
            "other_pct": float(other_pct),
        })

    rows.sort(
        key=lambda item: (
            -item["score"],
            -item["cluster_pct"],
            -item["cluster_count"],
            item["ingredient"],
        )
    )

    return rows[:top_n]


def representative_recipes(
    vectors: np.ndarray,
    cluster_indices: np.ndarray,
    metadata: pd.DataFrame,
    *,
    n_representatives: int,
) -> list[dict[str, Any]]:
    """Return recipes closest to cluster centroid."""
    sub_vectors = vectors[cluster_indices]
    centroid = sub_vectors.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    similarities = sub_vectors @ centroid
    order = np.argsort(-similarities)[:n_representatives]

    representatives: list[dict[str, Any]] = []
    for relative_index in order:
        absolute_index = int(cluster_indices[relative_index])
        row = metadata.iloc[absolute_index]

        title = safe_str(row.get("title"), safe_str(row.get("recipe_id")))
        representatives.append({
            "recipe_id": str(row["recipe_id"]),
            "title": title,
            "source_id": row.get("source_id"),
            "period": row.get("period_derived"),
            "similarity_to_centroid": round(float(similarities[relative_index]), 6),
        })

    return representatives


def inspect_clusters(
    *,
    vectors: np.ndarray,
    ids: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    recipe_to_ingredients: dict[str, list[str]],
    ingredient_labels: dict[str, str],
    top_ingredients: int,
    n_representatives: int,
    top_distribution_items: int,
    min_cluster_ingredient_pct: float,
    distinctiveness_smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = metadata.copy()
    metadata["cluster"] = labels

    all_recipe_ids = [str(recipe_id) for recipe_id in ids]
    all_indices = np.arange(len(ids))

    cluster_rows: list[dict[str, Any]] = []
    ingredient_rows: list[dict[str, Any]] = []

    for cluster_id in sorted(set(labels)):
        cluster_indices = all_indices[labels == cluster_id]
        other_indices = all_indices[labels != cluster_id]

        cluster_recipe_ids = [str(ids[index]) for index in cluster_indices]
        other_recipe_ids = [str(ids[index]) for index in other_indices]

        sub = metadata.iloc[cluster_indices]

        distinctive = distinctive_ingredients(
            cluster_recipe_ids=cluster_recipe_ids,
            other_recipe_ids=other_recipe_ids,
            recipe_to_ingredients=recipe_to_ingredients,
            ingredient_labels=ingredient_labels,
            top_n=top_ingredients,
            min_cluster_pct=min_cluster_ingredient_pct,
            smoothing=distinctiveness_smoothing,
        )

        reps = representative_recipes(
            vectors,
            cluster_indices,
            metadata,
            n_representatives=n_representatives,
        )

        for rank, item in enumerate(distinctive, start=1):
            ingredient_rows.append({
                "cluster": int(cluster_id),
                "rank": rank,
                **item,
            })

        cluster_rows.append({
            "cluster": int(cluster_id),
            "n": int(len(cluster_indices)),
            "top_languages": top_counts_str(sub["source_language"], top_distribution_items),
            "top_sources": top_counts_str(sub["source_id"], top_distribution_items),
            "top_periods": top_counts_str(sub["period_derived"], top_distribution_items),
            "top_places": top_counts_str(sub["source_place"], top_distribution_items),
            "distinctive_ingredients": "; ".join(
                f"{item['ingredient']}({item['score']:.2f}, {item['cluster_count']}/{len(cluster_recipe_ids)})"
                for item in distinctive
            ),
            "representative_titles": " | ".join(item["title"] for item in reps),
            "representative_recipe_ids": ", ".join(item["recipe_id"] for item in reps),
            "representatives_json": json.dumps(reps, ensure_ascii=False),
            "languages_json": json.dumps(top_counts(sub["source_language"], top_distribution_items), ensure_ascii=False),
            "sources_json": json.dumps(top_counts(sub["source_id"], top_distribution_items), ensure_ascii=False),
            "periods_json": json.dumps(top_counts(sub["period_derived"], top_distribution_items), ensure_ascii=False),
            "places_json": json.dumps(top_counts(sub["source_place"], top_distribution_items), ensure_ascii=False),
        })

    clusters_df = pd.DataFrame(cluster_rows).sort_values(
        by=["n", "cluster"],
        ascending=[False, True],
        kind="mergesort",
    )
    ingredients_df = pd.DataFrame(ingredient_rows).sort_values(
        by=["cluster", "rank"],
        ascending=[True, True],
        kind="mergesort",
    )

    return clusters_df, ingredients_df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def print_preview(clusters_df: pd.DataFrame, *, n_preview: int) -> None:
    log.info("Top %d clusters by size:", min(n_preview, len(clusters_df)))

    for _, row in clusters_df.head(n_preview).iterrows():
        log.info("=" * 72)
        log.info("CLUSTER %d (n=%d)", row["cluster"], row["n"])
        log.info("  languages: %s", row["top_languages"])
        log.info("  sources:   %s", row["top_sources"])
        log.info("  periods:   %s", row["top_periods"])
        log.info("  places:    %s", row["top_places"])
        log.info("  distinctive ingredients: %s", row["distinctive_ingredients"])
        log.info("  representative titles: %s", row["representative_titles"])


# ---------------------------------------------------------------------------
# CLI
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
        help=f"Canonical graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Layer 2 embeddings NPZ. Default: {DEFAULT_EMBEDDINGS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="k_NN value.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Optional Leiden RBConfiguration resolution. Omit for modularity partition.",
    )
    parser.add_argument(
        "--top-ingredients",
        type=int,
        default=DEFAULT_TOP_INGREDIENTS,
        help="Number of distinctive ingredients per cluster.",
    )
    parser.add_argument(
        "--n-representatives",
        type=int,
        default=DEFAULT_N_REPRESENTATIVES,
        help="Number of representative recipes per cluster.",
    )
    parser.add_argument(
        "--top-distribution-items",
        type=int,
        default=3,
        help="Number of top languages/sources/periods/places to show.",
    )
    parser.add_argument(
        "--min-cluster-ingredient-pct",
        type=float,
        default=DEFAULT_MIN_CLUSTER_INGREDIENT_PCT,
        help="Minimum ingredient prevalence inside a cluster to be reported.",
    )
    parser.add_argument(
        "--distinctiveness-smoothing",
        type=float,
        default=0.001,
        help="Smoothing added to outside-cluster prevalence.",
    )
    parser.add_argument(
        "--preview-clusters",
        type=int,
        default=8,
        help="Number of largest clusters to print in log preview.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
    )
    parser.add_argument(
        "--allow-pickle-npz",
        action="store_true",
        help=(
            "Allow pickle when loading NPZ. Avoid unless your old NPZ requires "
            "object arrays."
        ),
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

    if args.k < 1:
        log.error("--k must be >= 1")
        return 2
    if args.resolution is not None and args.resolution <= 0:
        log.error("--resolution must be positive")
        return 2
    if args.top_ingredients < 1:
        log.error("--top-ingredients must be >= 1")
        return 2
    if args.n_representatives < 1:
        log.error("--n-representatives must be >= 1")
        return 2
    if args.top_distribution_items < 1:
        log.error("--top-distribution-items must be >= 1")
        return 2
    if not (0 <= args.min_cluster_ingredient_pct <= 1):
        log.error("--min-cluster-ingredient-pct must be in [0, 1]")
        return 2
    if args.distinctiveness_smoothing < 0:
        log.error("--distinctiveness-smoothing must be >= 0")
        return 2
    if args.preview_clusters < 0:
        log.error("--preview-clusters must be >= 0")
        return 2

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        _, vectors, ids, metadata, recipe_to_ingredients, ingredient_labels = load_data(
            graph_path=args.graph,
            embeddings_path=args.embeddings,
            allow_pickle_npz=args.allow_pickle_npz,
        )

        knn_graph, k_effective = build_knn_graph(vectors, k=args.k)
        labels, modularity = run_leiden(
            knn_graph,
            random_state=args.random_state,
            resolution=args.resolution,
        )

        metadata_with_clusters = metadata.copy()
        metadata_with_clusters["cluster"] = labels
        metadata_with_clusters["k_NN_used"] = args.k
        metadata_with_clusters["k_NN_effective"] = k_effective

        clusters_df, ingredients_df = inspect_clusters(
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            labels=labels,
            recipe_to_ingredients=recipe_to_ingredients,
            ingredient_labels=ingredient_labels,
            top_ingredients=args.top_ingredients,
            n_representatives=args.n_representatives,
            top_distribution_items=args.top_distribution_items,
            min_cluster_ingredient_pct=args.min_cluster_ingredient_pct,
            distinctiveness_smoothing=args.distinctiveness_smoothing,
        )

        inspection_path = args.output_dir / f"cluster_inspection_k{args.k}.csv"
        assignments_path = args.output_dir / f"cluster_assignments_k{args.k}.csv"
        ingredients_path = args.output_dir / f"cluster_distinctive_ingredients_k{args.k}.csv"
        summary_path = args.output_dir / f"cluster_inspection_k{args.k}_summary.json"

        write_csv(clusters_df, inspection_path)
        write_csv(metadata_with_clusters, assignments_path)
        write_csv(ingredients_df, ingredients_path)

        size_counts = Counter(labels.tolist())
        summary = {
            "inputs": {
                "graph": str(args.graph),
                "embeddings": str(args.embeddings),
            },
            "outputs": {
                "cluster_inspection_csv": str(inspection_path),
                "cluster_assignments_csv": str(assignments_path),
                "distinctive_ingredients_csv": str(ingredients_path),
                "summary_json": str(summary_path),
            },
            "parameters": {
                "k": args.k,
                "k_effective": k_effective,
                "resolution": args.resolution,
                "top_ingredients": args.top_ingredients,
                "n_representatives": args.n_representatives,
                "min_cluster_ingredient_pct": args.min_cluster_ingredient_pct,
                "distinctiveness_smoothing": args.distinctiveness_smoothing,
                "random_state": args.random_state,
                "allow_pickle_npz": args.allow_pickle_npz,
            },
            "n_recipes": int(len(ids)),
            "embedding_dim": int(vectors.shape[1]),
            "n_clusters": int(len(size_counts)),
            "modularity_Q": round(float(modularity), 6),
            "cluster_size_summary": {
                "largest": int(max(size_counts.values())),
                "smallest": int(min(size_counts.values())),
                "median": float(np.median(list(size_counts.values()))),
            },
        }

        write_json(summary_path, summary)

        if args.preview_clusters:
            print_preview(clusters_df, n_preview=args.preview_clusters)

        log.info("Done.")
        return 0

    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.PickleError,
        json.JSONDecodeError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
