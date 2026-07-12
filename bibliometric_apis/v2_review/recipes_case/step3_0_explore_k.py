"""
====================

Explore k-NN parameters for recipe clustering.

Before committing to a recipe clustering, this script sweeps over several k
values for the recipe-embedding k-NN graph and reports clustering metrics. The
goal is to make the k choice empirical rather than arbitrary.

For each k:

  1. Build a weighted undirected k-NN graph on recipe embeddings.
  2. Run Leiden community detection by default.
  3. Measure:
       - number of communities
       - modularity Q
       - number of connected components
       - cosine silhouette score on a random sample
       - cluster-size summary
       - known-source separation checks

Default inputs
--------------

  data/graph_step2_canonical.gpickle
  data/layer_2_embeddings.npz

Default outputs
---------------

  data/k_exploration_metrics.csv
  data/k_exploration_clusters.csv
  data/k_exploration_cluster_summary.csv
  data/k_exploration_stats.json
  data/k_exploration_pca.png

Usage
-----

  python step3_0_explore_k.py

  python step3_0_explore_k.py --k-values 10 20 30 50

  python step3_0_explore_k.py --no-plot

  python step3_0_explore_k.py \
    --known-pair apicius:corema \
    --known-pair cuina_catalana:corema

Dependencies
------------

Required:

  pip install numpy pandas scikit-learn igraph leidenalg

Optional for plotting:

  pip install matplotlib


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step2_canonical.gpickle")
DEFAULT_EMBEDDINGS = Path("data/layer_2_embeddings.npz")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_K_VALUES = [10, 15, 20, 30, 50, 100]
DEFAULT_KNOWN_PAIRS = [
    "apicius:corema",
    "fundacio_alicia:corema",
    "cuina_catalana:corema",
]

DEFAULT_SILHOUETTE_SAMPLE_SIZE = 1500
DEFAULT_RANDOM_STATE = 42

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step3_0_explore_k")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    """Load a trusted local NetworkX graph."""
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

        recipe_ids = npz["recipes_node_ids"].astype(str)
        vectors = np.asarray(npz["recipes_vectors"], dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"recipes_vectors must be 2D, got shape {vectors.shape}")

    if len(recipe_ids) != vectors.shape[0]:
        raise ValueError(
            f"recipes_node_ids length {len(recipe_ids)} does not match "
            f"recipes_vectors rows {vectors.shape[0]}"
        )

    if len(recipe_ids) == 0:
        raise ValueError("No recipe embeddings found")

    if not np.isfinite(vectors).all():
        raise ValueError("Recipe embedding matrix contains NaN or Inf values")

    log.info("Recipe embeddings: %d rows, dim=%d", vectors.shape[0], vectors.shape[1])
    return recipe_ids, vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    n_zero = int((norms[:, 0] < 1e-12).sum())
    if n_zero:
        raise ValueError(f"Recipe embedding matrix contains {n_zero} near-zero vectors")
    return vectors / np.maximum(norms, 1e-12)


def load_recipe_data(
    *,
    graph_path: Path,
    embeddings_path: Path,
    allow_pickle_npz: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return normalized vectors, recipe IDs, and metadata in the same order."""
    graph = load_graph(graph_path)
    all_ids, all_vectors = load_recipe_embeddings(embeddings_path, allow_pickle=allow_pickle_npz)

    in_graph = np.array([str(node_id) in graph for node_id in all_ids], dtype=bool)
    ids = all_ids[in_graph]
    vectors = all_vectors[in_graph]

    if len(ids) == 0:
        raise ValueError("No recipe embeddings match nodes in the graph")

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

    log.info("%d recipes with embeddings present in graph", len(ids))
    log.info("Metadata loaded for %d recipes", len(metadata))

    return vectors, ids, metadata


# ---------------------------------------------------------------------------
# Graph building and clustering
# ---------------------------------------------------------------------------

def build_knn_graph(
    vectors: np.ndarray,
    *,
    k: int,
):
    """Build a weighted undirected igraph k-NN graph.

    Edge weight = cosine similarity. Mutual directed k-NN links are collapsed by
    keeping the maximum similarity, not by summing, so weights remain in [0, 1].
    """
    try:
        import igraph as ig
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "k-NN graph construction requires scikit-learn and igraph: "
            "pip install scikit-learn igraph"
        ) from exc

    n = vectors.shape[0]
    if n < 2:
        raise ValueError("Need at least two recipes to build a k-NN graph")

    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"Invalid k={k} for n={n}")

    if k_eff != k:
        log.warning("k=%d is too large for n=%d; using k=%d", k, n, k_eff)

    log.info("Building k-NN graph with k=%d on %d vectors", k_eff, n)
    start = time.time()

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    log.info("k-NN search completed in %.1fs", time.time() - start)

    edge_weight_by_pair: dict[tuple[int, int], float] = {}

    for i in range(n):
        for neighbour_rank in range(1, k_eff + 1):  # skip self at rank 0
            j = int(indices[i, neighbour_rank])
            if i == j:
                continue

            similarity = 1.0 - float(distances[i, neighbour_rank])
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

    log.info("igraph graph: %d nodes, %d edges", graph.vcount(), graph.ecount())
    return graph


def run_leiden(
    graph,
    *,
    random_state: int,
    resolution: float | None,
) -> tuple[list[int], float]:
    """Run Leiden community detection."""
    try:
        import leidenalg
    except ImportError as exc:
        raise RuntimeError("Leiden clustering requires leidenalg: pip install leidenalg") from exc

    log.info("Running Leiden community detection")
    start = time.time()

    if resolution is None:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=random_state,
        )
    else:
        # RBConfiguration allows explicit resolution control.
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )

    membership = [int(x) for x in partition.membership]
    modularity = float(partition.modularity)

    log.info(
        "Leiden completed in %.1fs: %d clusters, Q=%.4f",
        time.time() - start,
        len(set(membership)),
        modularity,
    )

    return membership, modularity


def connected_components_count(graph) -> int:
    try:
        return int(len(graph.connected_components()))
    except TypeError:
        return int(len(graph.connected_components(mode="weak")))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def silhouette_sample(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> float | None:
    """Cosine silhouette on a sample for tractability."""
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("Silhouette scoring requires scikit-learn") from exc

    n = len(vectors)
    if n < 3:
        return None

    rng = np.random.default_rng(random_state)
    if n > sample_size:
        sample_idx = rng.choice(n, sample_size, replace=False)
    else:
        sample_idx = np.arange(n)

    sampled_labels = labels[sample_idx]
    n_labels = len(set(sampled_labels))

    # sklearn requires 2 <= n_labels <= n_samples - 1.
    if n_labels < 2 or n_labels >= len(sample_idx):
        return None

    return float(silhouette_score(vectors[sample_idx], sampled_labels, metric="cosine"))


def parse_known_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --known-pair {item!r}; expected source_a:source_b")
        left, _, right = item.partition(":")
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid --known-pair {item!r}; expected source_a:source_b")
        pairs.append((left, right))

    return pairs


def source_pair_cluster_stats(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    source_a: str,
    source_b: str,
) -> dict[str, Any] | None:
    df = metadata.copy()
    df["cluster"] = labels

    a = df[df["source_id"] == source_a]
    b = df[df["source_id"] == source_b]

    if a.empty or b.empty:
        return None

    clusters_a = set(a["cluster"])
    clusters_b = set(b["cluster"])
    shared = clusters_a & clusters_b

    # Approximate pair-level separation without materializing all pairs.
    total_pairs = len(a) * len(b)
    same_cluster_pairs = 0
    for cluster_id in shared:
        same_cluster_pairs += int((a["cluster"] == cluster_id).sum()) * int((b["cluster"] == cluster_id).sum())

    different_cluster_pairs = total_pairs - same_cluster_pairs
    different_fraction = different_cluster_pairs / total_pairs if total_pairs else None

    return {
        "source_a": source_a,
        "source_b": source_b,
        "a_size": int(len(a)),
        "b_size": int(len(b)),
        "a_clusters": int(len(clusters_a)),
        "b_clusters": int(len(clusters_b)),
        "shared_clusters": int(len(shared)),
        "pair_different_cluster_fraction": round(float(different_fraction), 6)
        if different_fraction is not None
        else None,
    }


def known_separation_checks(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    known_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for source_a, source_b in known_pairs:
        key = f"{source_a}_vs_{source_b}"
        out[key] = source_pair_cluster_stats(metadata, labels, source_a, source_b)

    return out


def cluster_summary_table(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = metadata.copy()
    df["cluster"] = labels

    rows: list[dict[str, Any]] = []

    for cluster_id, sub in df.groupby("cluster"):
        source_counts = sub["source_id"].value_counts(dropna=True)
        language_counts = sub["source_language"].value_counts(dropna=True)
        period_counts = sub["period_derived"].value_counts(dropna=True)

        rows.append({
            "cluster": int(cluster_id),
            "n": int(len(sub)),
            "top_source": source_counts.index[0] if len(source_counts) else None,
            "top_source_n": int(source_counts.iloc[0]) if len(source_counts) else 0,
            "top_language": language_counts.index[0] if len(language_counts) else None,
            "top_language_n": int(language_counts.iloc[0]) if len(language_counts) else 0,
            "top_period": period_counts.index[0] if len(period_counts) else None,
            "top_period_n": int(period_counts.iloc[0]) if len(period_counts) else 0,
            "n_distinct_sources": int(sub["source_id"].nunique(dropna=True)),
            "n_distinct_languages": int(sub["source_language"].nunique(dropna=True)),
            "n_distinct_periods": int(sub["period_derived"].nunique(dropna=True)),
        })

    return pd.DataFrame(rows).sort_values(
        by=["n", "cluster"],
        ascending=[False, True],
        kind="mergesort",
    )


def choose_best_k(metrics_df: pd.DataFrame) -> int:
    """Choose suggested k using a simple transparent heuristic."""
    candidates = metrics_df[
        (metrics_df["silhouette_cosine"].fillna(-1) > 0)
        & (metrics_df["n_clusters"].between(5, 50))
        & (metrics_df["n_connected_components"] <= 2)
    ]

    if candidates.empty:
        candidates = metrics_df[
            (metrics_df["silhouette_cosine"].fillna(-1) > 0)
            & (metrics_df["n_clusters"].between(5, 50))
        ]

    if candidates.empty:
        return int(metrics_df.sort_values("modularity_Q", ascending=False).iloc[0]["k_NN"])

    return int(
        candidates.sort_values(
            by=["modularity_Q", "silhouette_cosine"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]["k_NN"]
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca_grid(
    vectors: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    out_path: Path,
    *,
    random_state: int,
    point_size: float,
    alpha: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError("PCA plot requires matplotlib and scikit-learn") from exc

    log.info("Computing 2D PCA for visualization")
    pca = PCA(n_components=2, random_state=random_state)
    vectors_2d = pca.fit_transform(vectors)

    log.info(
        "PCA explained variance: %.4f, %.4f",
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
    )

    n_panels = len(labels_by_k)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_array = np.array(axes).reshape(-1)

    for ax, (k, labels) in zip(axes_array, sorted(labels_by_k.items())):
        n_clusters = len(set(labels))
        ax.scatter(
            vectors_2d[:, 0],
            vectors_2d[:, 1],
            c=labels,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        ax.set_title(f"k_NN={k} ({n_clusters} clusters)")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_array[len(labels_by_k):]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved PCA grid: %s", out_path)


# ---------------------------------------------------------------------------
# Output helpers
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


def jsonable_metrics(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in metrics_rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> int:
    vectors, ids, metadata = load_recipe_data(
        graph_path=args.graph,
        embeddings_path=args.embeddings,
        allow_pickle_npz=args.allow_pickle_npz,
    )

    if len(args.k_values) != len(set(args.k_values)):
        log.warning("Duplicate k values were provided; duplicates will be ignored")

    k_values = sorted(set(args.k_values))
    known_pairs = parse_known_pairs(args.known_pair)

    labels_by_k: dict[int, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = {}

    metrics_list: list[dict[str, Any]] = []

    for k in k_values:
        if k < 1:
            raise ValueError(f"Invalid k={k}; k must be >= 1")

        log.info("=" * 72)
        log.info("k_NN = %d", k)
        log.info("=" * 72)

        graph = build_knn_graph(vectors, k=k)
        n_components = connected_components_count(graph)
        membership, modularity = run_leiden(
            graph,
            random_state=args.random_state,
            resolution=args.resolution,
        )
        labels = np.asarray(membership, dtype=int)

        silhouette = silhouette_sample(
            vectors,
            labels,
            sample_size=args.silhouette_sample_size,
            random_state=args.random_state,
        )

        size_counts = Counter(labels.tolist())
        top_sizes = sorted(size_counts.values(), reverse=True)[:5]
        separation = known_separation_checks(metadata, labels, known_pairs)

        row: dict[str, Any] = {
            "k_NN": int(k),
            "k_effective": int(min(k, len(vectors) - 1)),
            "n_clusters": int(len(size_counts)),
            "modularity_Q": round(float(modularity), 6),
            "silhouette_cosine": round(float(silhouette), 6) if silhouette is not None else None,
            "n_connected_components": int(n_components),
            "n_edges": int(graph.ecount()),
            "largest_cluster_size": int(max(size_counts.values())),
            "smallest_cluster_size": int(min(size_counts.values())),
            "top5_cluster_sizes": json.dumps(top_sizes),
        }

        for pair_key, stats in separation.items():
            row[f"{pair_key}_shared_clusters"] = None if stats is None else stats["shared_clusters"]
            row[f"{pair_key}_different_fraction"] = None if stats is None else stats["pair_different_cluster_fraction"]

        metrics_list.append(row)
        labels_by_k[k] = labels

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = args.output_dir / "k_exploration_metrics.csv"
    write_csv(metrics_df, metrics_path)

    log.info("\n%s", metrics_df.to_string(index=False))

    best_k = choose_best_k(metrics_df)
    log.info("Suggested best k_NN: %d", best_k)

    best_labels = labels_by_k[best_k]

    clusters_df = metadata.copy()
    clusters_df["cluster"] = best_labels
    clusters_df["k_NN_used"] = best_k
    clusters_path = args.output_dir / "k_exploration_clusters.csv"
    write_csv(clusters_df, clusters_path)

    cluster_summary = cluster_summary_table(metadata, best_labels)
    cluster_summary_path = args.output_dir / "k_exploration_cluster_summary.csv"
    write_csv(cluster_summary, cluster_summary_path)

    log.info(
        "\nCluster summary for k=%d:\n%s",
        best_k,
        cluster_summary.head(20).to_string(index=False),
    )

    plot_path = args.output_dir / "k_exploration_pca.png"
    if not args.no_plot:
        try:
            plot_pca_grid(
                vectors,
                labels_by_k,
                plot_path,
                random_state=args.random_state,
                point_size=args.plot_point_size,
                alpha=args.plot_alpha,
            )
        except RuntimeError as exc:
            log.warning("%s; skipping PCA plot", exc)

    stats = {
        "inputs": {
            "graph": str(args.graph),
            "embeddings": str(args.embeddings),
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "clusters_csv": str(clusters_path),
            "cluster_summary_csv": str(cluster_summary_path),
            "pca_png": None if args.no_plot else str(plot_path),
            "stats_json": str(args.output_dir / "k_exploration_stats.json"),
        },
        "parameters": {
            "k_values_swept": k_values,
            "resolution": args.resolution,
            "random_state": args.random_state,
            "silhouette_sample_size": args.silhouette_sample_size,
            "known_pairs": [f"{a}:{b}" for a, b in known_pairs],
            "allow_pickle_npz": args.allow_pickle_npz,
        },
        "suggested_best_k": int(best_k),
        "n_recipes": int(len(ids)),
        "embedding_dim": int(vectors.shape[1]),
        "metrics": jsonable_metrics(metrics_list),
        "known_separation_best_k": known_separation_checks(metadata, best_labels, known_pairs),
        "selection_heuristic": (
            "Prefer highest modularity among runs with positive silhouette, "
            "5-50 clusters, and <=2 connected components; otherwise relax "
            "connected-component criterion; otherwise highest modularity."
        ),
    }

    stats_path = args.output_dir / "k_exploration_stats.json"
    write_json(stats_path, stats)

    log.info("Done.")
    return 0


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
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="k values to sweep.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help=(
            "Optional Leiden RBConfiguration resolution. "
            "Omit to use modularity partition, matching the original script."
        ),
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE_SIZE,
        help="Sample size for cosine silhouette calculation.",
    )
    parser.add_argument(
        "--known-pair",
        action="append",
        default=DEFAULT_KNOWN_PAIRS,
        help="Known-distinct source pair as source_a:source_b. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-known-pairs",
        action="store_true",
        help="Ignore built-in known source-pair checks.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PCA plot.",
    )
    parser.add_argument(
        "--plot-point-size",
        type=float,
        default=4.0,
        help="PCA scatter point size.",
    )
    parser.add_argument(
        "--plot-alpha",
        type=float,
        default=0.6,
        help="PCA scatter alpha.",
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

    args = parser.parse_args(argv)

    if args.no_default_known_pairs:
        args.known_pair = [
            pair for pair in args.known_pair
            if pair not in DEFAULT_KNOWN_PAIRS
        ]

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.silhouette_sample_size < 2:
        log.error("--silhouette-sample-size must be >= 2")
        return 2
    if not args.k_values:
        log.error("--k-values must contain at least one integer")
        return 2
    if args.resolution is not None and args.resolution <= 0:
        log.error("--resolution must be positive")
        return 2
    if args.plot_point_size <= 0:
        log.error("--plot-point-size must be positive")
        return 2
    if not (0 < args.plot_alpha <= 1):
        log.error("--plot-alpha must be in (0, 1]")
        return 2

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return run_sweep(args)

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
