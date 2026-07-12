"""
========================

Mitigate source bias in recipe embeddings through source-centering.

Problem
-------

Raw recipe embeddings can cluster by source_id rather than by culinary content.
For example, a cluster may be dominated by one manuscript, translator, language,
or editorial style. That is useful to diagnose, but bad if the goal is a
culinary-category clustering.

Method
------

For each source_id with at least --min-source recipes:

    centroid_s = mean(recipe vectors for source s)

For each recipe r in source s:

    v_r_debiased = v_r - alpha * centroid_s

Then all recipe vectors are L2-renormalized.

  alpha=1.0  full source-centering
  alpha=0.5  softer source-centering
  alpha=0.0  no debiasing, useful as a control

Sources smaller than --min-source are left unchanged because their centroid is
too noisy.

Default inputs
--------------

  data/layer_2_embeddings.npz
  data/graph_step2_canonical.gpickle

Default outputs
---------------

  data/layer_2_embeddings_debiased.npz
  data/source_centering_diagnostics.json
  data/source_centering_source_diagnostics.csv
  data/source_centering_per_cluster_comparison.csv
  data/k_exploration_debiased.csv
  data/source_centering_cluster_assignments.csv

Usage
-----

  python step3_1_source_debias.py

  python step3_1_source_debias.py --alpha 0.5

  python step3_1_source_debias.py --min-source 30 --alpha 0.7

  python step3_1_source_debias.py --dry-run

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS_IN = Path("data/layer_2_embeddings.npz")
DEFAULT_EMBEDDINGS_OUT = Path("data/layer_2_embeddings_debiased.npz")
DEFAULT_GRAPH = Path("data/graph_step2_canonical.gpickle")
DEFAULT_DATA_DIR = Path("data")

DEFAULT_ALPHA = 1.0
DEFAULT_MIN_SOURCE = 20
DEFAULT_K = 30
DEFAULT_RANDOM_STATE = 42

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step3_1_source_debias")


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


def load_npz(path: Path, *, allow_pickle: bool) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings NPZ not found: {path}")

    log.info("Loading embeddings: %s", path)
    with np.load(path, allow_pickle=allow_pickle) as npz:
        arrays = {key: npz[key] for key in npz.files}

    required = {"recipes_node_ids", "recipes_vectors"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Embeddings NPZ missing required arrays: {sorted(missing)}")

    recipe_ids = arrays["recipes_node_ids"].astype(str)
    recipe_vectors = np.asarray(arrays["recipes_vectors"], dtype=np.float32)

    if recipe_vectors.ndim != 2:
        raise ValueError(f"recipes_vectors must be 2D, got shape {recipe_vectors.shape}")

    if len(recipe_ids) != recipe_vectors.shape[0]:
        raise ValueError(
            f"recipes_node_ids length {len(recipe_ids)} does not match "
            f"recipes_vectors rows {recipe_vectors.shape[0]}"
        )

    if not np.isfinite(recipe_vectors).all():
        raise ValueError("recipes_vectors contains NaN or Inf values")

    log.info("Recipe embeddings: %d rows, dim=%d", recipe_vectors.shape[0], recipe_vectors.shape[1])
    return arrays


def normalize_vectors(vectors: np.ndarray) -> tuple[np.ndarray, int]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    near_zero = norms[:, 0] < 1e-12
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized, int(near_zero.sum())


def recipe_metadata(graph: Any, recipe_ids: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Return in-graph mask, metadata table, and sources aligned to in-graph IDs."""
    in_graph = np.array([str(node_id) in graph for node_id in recipe_ids], dtype=bool)
    ids_in_graph = recipe_ids[in_graph]

    rows: list[dict[str, Any]] = []
    sources: list[str] = []

    for node_id in ids_in_graph:
        attrs = graph.nodes[str(node_id)]
        source_id = str(attrs.get("source_id") or "")
        sources.append(source_id)
        rows.append({
            "recipe_id": str(node_id),
            "source_id": source_id,
            "source_language": attrs.get("source_language"),
            "source_place": attrs.get("source_place"),
            "period_derived": attrs.get("period_derived"),
            "source_year": attrs.get("source_year"),
            "title": attrs.get("title"),
        })

    metadata = pd.DataFrame(rows)

    if metadata.empty:
        raise ValueError("No recipe IDs from embeddings are present in the graph")

    log.info(
        "%d/%d recipe embedding rows are present in graph; %d unique sources",
        int(in_graph.sum()),
        len(recipe_ids),
        metadata["source_id"].nunique(dropna=False),
    )

    return in_graph, metadata, np.asarray(sources, dtype=object)


# ---------------------------------------------------------------------------
# Clustering and source-dominance metrics
# ---------------------------------------------------------------------------

def build_knn_graph(vectors: np.ndarray, *, k: int):
    """Build weighted undirected igraph k-NN graph.

    Mutual directed k-NN links are collapsed by keeping the max similarity so
    weights remain in [0, 1].
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
        raise ValueError("Need at least two recipe vectors to cluster")

    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"Invalid k={k} for n={n}")

    if k_eff != k:
        log.warning("k=%d is too large for n=%d; using k=%d", k, n, k_eff)

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
        "Built k-NN graph k=%d in %.1fs: %d nodes, %d edges",
        k_eff,
        time.time() - start,
        graph.vcount(),
        graph.ecount(),
    )

    return graph, k_eff


def run_leiden(
    vectors: np.ndarray,
    *,
    k: int,
    random_state: int,
    resolution: float | None,
) -> tuple[np.ndarray, float, int, int]:
    """Cluster recipe vectors and return labels, modularity, edge count, k_eff."""
    try:
        import leidenalg
    except ImportError as exc:
        raise RuntimeError("Requires leidenalg: pip install leidenalg") from exc

    graph, k_eff = build_knn_graph(vectors, k=k)

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
        "Leiden: %d clusters, Q=%.4f",
        len(set(labels)),
        modularity,
    )

    return labels, modularity, int(graph.ecount()), k_eff


def source_dominance(labels: np.ndarray, sources: np.ndarray) -> dict[int, dict[str, Any]]:
    """Compute per-cluster source-dominance statistics."""
    out: dict[int, dict[str, Any]] = {}

    for cluster_id in sorted(set(labels)):
        cluster_sources = sources[labels == cluster_id]
        if len(cluster_sources) == 0:
            continue

        counts = Counter(str(source) for source in cluster_sources)
        top_source, top_count = counts.most_common(1)[0]

        out[int(cluster_id)] = {
            "n": int(len(cluster_sources)),
            "top_source": str(top_source),
            "top_count": int(top_count),
            "top_pct": round(100 * top_count / len(cluster_sources), 3),
            "n_distinct_sources": int(len(counts)),
        }

    return out


def dominance_summary(dominance: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not dominance:
        return {
            "n_clusters": 0,
            "median_top_pct": None,
            "mean_top_pct": None,
            "n_clusters_above_80pct": 0,
            "n_clusters_above_50pct": 0,
        }

    top_pcts = [float(item["top_pct"]) for item in dominance.values()]

    return {
        "n_clusters": int(len(dominance)),
        "median_top_pct": round(float(np.median(top_pcts)), 3),
        "mean_top_pct": round(float(np.mean(top_pcts)), 3),
        "n_clusters_above_80pct": int(sum(1 for pct in top_pcts if pct >= 80)),
        "n_clusters_above_50pct": int(sum(1 for pct in top_pcts if pct >= 50)),
    }


def clustering_metrics(
    vectors: np.ndarray,
    sources: np.ndarray,
    metadata: pd.DataFrame,
    *,
    variant: str,
    k: int,
    random_state: int,
    resolution: float | None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], np.ndarray]:
    labels, modularity, n_edges, k_eff = run_leiden(
        vectors,
        k=k,
        random_state=random_state,
        resolution=resolution,
    )

    dominance = source_dominance(labels, sources)
    dominance_stats = dominance_summary(dominance)

    metrics = {
        "variant": variant,
        "k_NN": int(k),
        "k_effective": int(k_eff),
        "resolution": resolution,
        "n_recipes": int(len(vectors)),
        "n_edges": int(n_edges),
        "modularity_Q": round(float(modularity), 6),
        **dominance_stats,
    }

    return metrics, dominance, labels


# ---------------------------------------------------------------------------
# Debiasing
# ---------------------------------------------------------------------------

def source_center_vectors(
    vectors: np.ndarray,
    sources: np.ndarray,
    *,
    alpha: float,
    min_source: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Apply source-centering to normalized recipe vectors."""
    debiased = vectors.copy()
    source_sizes = Counter(str(source) for source in sources)

    source_rows: list[dict[str, Any]] = []
    centered_sources = 0
    skipped_sources = 0
    zero_after_center = 0

    for source_id, count in sorted(source_sizes.items(), key=lambda item: (-item[1], item[0])):
        mask = sources == source_id

        if not source_id:
            source_rows.append({
                "source_id": source_id,
                "n_recipes": int(count),
                "centered": False,
                "reason": "empty_source_id",
                "centroid_norm": None,
                "mean_cosine_to_centroid_before": None,
                "mean_cosine_to_centroid_after": None,
                "zero_after_center": 0,
            })
            skipped_sources += 1
            continue

        if count < min_source:
            source_rows.append({
                "source_id": source_id,
                "n_recipes": int(count),
                "centered": False,
                "reason": f"n<{min_source}",
                "centroid_norm": None,
                "mean_cosine_to_centroid_before": None,
                "mean_cosine_to_centroid_after": None,
                "zero_after_center": 0,
            })
            skipped_sources += 1
            continue

        centroid = vectors[mask].mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))

        if centroid_norm < 1e-12:
            source_rows.append({
                "source_id": source_id,
                "n_recipes": int(count),
                "centered": False,
                "reason": "near_zero_centroid",
                "centroid_norm": centroid_norm,
                "mean_cosine_to_centroid_before": None,
                "mean_cosine_to_centroid_after": None,
                "zero_after_center": 0,
            })
            skipped_sources += 1
            continue

        centroid_unit = centroid / centroid_norm
        before_alignment = float(np.mean(vectors[mask] @ centroid_unit))

        adjusted = vectors[mask] - alpha * centroid
        adjusted_norms = np.linalg.norm(adjusted, axis=1, keepdims=True)
        zero_mask = adjusted_norms[:, 0] < 1e-12

        n_zero = int(zero_mask.sum())
        zero_after_center += n_zero

        if n_zero:
            # Preserve original normalized vector where centering degenerates.
            adjusted[zero_mask] = vectors[mask][zero_mask]
            adjusted_norms = np.linalg.norm(adjusted, axis=1, keepdims=True)

        adjusted = adjusted / np.maximum(adjusted_norms, 1e-12)
        debiased[mask] = adjusted

        after_alignment = float(np.mean(debiased[mask] @ centroid_unit))

        source_rows.append({
            "source_id": source_id,
            "n_recipes": int(count),
            "centered": True,
            "reason": "",
            "centroid_norm": round(centroid_norm, 8),
            "mean_cosine_to_centroid_before": round(before_alignment, 8),
            "mean_cosine_to_centroid_after": round(after_alignment, 8),
            "zero_after_center": n_zero,
        })
        centered_sources += 1

    diagnostics = {
        "n_sources_total": int(len(source_sizes)),
        "n_sources_centered": int(centered_sources),
        "n_sources_skipped": int(skipped_sources),
        "zero_vectors_after_centering_fallback": int(zero_after_center),
    }

    return debiased.astype(np.float32), pd.DataFrame(source_rows), diagnostics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_debiased_npz(
    arrays: dict[str, np.ndarray],
    *,
    recipe_vectors_debiased_full: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_arrays = dict(arrays)
    save_arrays["recipes_vectors"] = recipe_vectors_debiased_full.astype(np.float32)

    np.savez_compressed(out_path, **save_arrays)
    log.info("Wrote debiased embeddings: %s", out_path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def per_cluster_comparison_rows(
    variant: str,
    dominance: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"variant": variant, "cluster": cluster_id, **info}
        for cluster_id, info in dominance.items()
    ]


def assignments_dataframe(
    metadata: pd.DataFrame,
    raw_labels: np.ndarray,
    debiased_labels: np.ndarray,
) -> pd.DataFrame:
    out = metadata.copy()
    out["cluster_raw"] = raw_labels
    out["cluster_debiased"] = debiased_labels
    out["cluster_changed"] = out["cluster_raw"] != out["cluster_debiased"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--embeddings-in",
        type=Path,
        default=DEFAULT_EMBEDDINGS_IN,
        help=f"Input embeddings NPZ. Default: {DEFAULT_EMBEDDINGS_IN}",
    )
    parser.add_argument(
        "--embeddings-out",
        type=Path,
        default=DEFAULT_EMBEDDINGS_OUT,
        help=f"Output debiased embeddings NPZ. Default: {DEFAULT_EMBEDDINGS_OUT}",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Canonical graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Fraction of source centroid to subtract. Usually 0.0-1.0.",
    )
    parser.add_argument(
        "--min-source",
        type=int,
        default=DEFAULT_MIN_SOURCE,
        help="Minimum recipes in a source for centering.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="k_NN value for raw/debiased clustering comparison.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Optional Leiden RBConfiguration resolution. Omit for modularity partition.",
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
        "--dry-run",
        action="store_true",
        help="Run diagnostics but do not write the debiased NPZ.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write diagnostics CSV/JSON.",
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

    if args.min_source < 2:
        log.error("--min-source must be >= 2")
        return 2
    if args.k < 1:
        log.error("--k must be >= 1")
        return 2
    if args.resolution is not None and args.resolution <= 0:
        log.error("--resolution must be positive")
        return 2
    if args.alpha < 0:
        log.error("--alpha must be >= 0")
        return 2
    if args.alpha > 1:
        log.warning("--alpha > 1.0 is allowed but aggressive; inspect diagnostics carefully.")

    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)

        arrays = load_npz(args.embeddings_in, allow_pickle=args.allow_pickle_npz)
        graph = load_graph(args.graph)

        all_recipe_ids = arrays["recipes_node_ids"].astype(str)
        all_recipe_vectors_raw = np.asarray(arrays["recipes_vectors"], dtype=np.float32)

        in_graph, metadata, sources = recipe_metadata(graph, all_recipe_ids)
        vectors_raw = all_recipe_vectors_raw[in_graph].astype(np.float32)
        vectors_raw_norm, n_zero_raw = normalize_vectors(vectors_raw)

        if n_zero_raw:
            raise ValueError(f"Raw recipe embeddings contain {n_zero_raw} near-zero vectors")

        log.info("Baseline clustering on raw embeddings")
        raw_metrics, raw_dominance, raw_labels = clustering_metrics(
            vectors_raw_norm,
            sources,
            metadata,
            variant="raw",
            k=args.k,
            random_state=args.random_state,
            resolution=args.resolution,
        )

        log.info(
            "Applying source-centering: alpha=%.3f, min_source=%d",
            args.alpha,
            args.min_source,
        )
        vectors_debiased, source_diag_df, centering_diag = source_center_vectors(
            vectors_raw_norm,
            sources,
            alpha=args.alpha,
            min_source=args.min_source,
        )

        log.info("Clustering on source-centered embeddings")
        debiased_metrics, debiased_dominance, debiased_labels = clustering_metrics(
            vectors_debiased,
            sources,
            metadata,
            variant="debiased",
            k=args.k,
            random_state=args.random_state,
            resolution=args.resolution,
        )

        metrics_df = pd.DataFrame([raw_metrics, debiased_metrics])
        metrics_path = args.data_dir / "k_exploration_debiased.csv"

        cluster_rows = (
            per_cluster_comparison_rows("raw", raw_dominance)
            + per_cluster_comparison_rows("debiased", debiased_dominance)
        )
        cluster_comparison_df = pd.DataFrame(cluster_rows)
        cluster_comparison_path = args.data_dir / "source_centering_per_cluster_comparison.csv"

        source_diag_path = args.data_dir / "source_centering_source_diagnostics.csv"
        assignments_path = args.data_dir / "source_centering_cluster_assignments.csv"
        diagnostics_path = args.data_dir / "source_centering_diagnostics.json"

        assignments_df = assignments_dataframe(metadata, raw_labels, debiased_labels)

        diagnostics = {
            "inputs": {
                "embeddings_in": str(args.embeddings_in),
                "graph": str(args.graph),
            },
            "outputs": {
                "embeddings_out": None if args.dry_run else str(args.embeddings_out),
                "diagnostics_json": str(diagnostics_path),
                "source_diagnostics_csv": str(source_diag_path),
                "per_cluster_comparison_csv": str(cluster_comparison_path),
                "k_exploration_debiased_csv": str(metrics_path),
                "cluster_assignments_csv": str(assignments_path),
            },
            "parameters": {
                "alpha": args.alpha,
                "min_source": args.min_source,
                "k": args.k,
                "resolution": args.resolution,
                "random_state": args.random_state,
                "allow_pickle_npz": args.allow_pickle_npz,
                "dry_run": args.dry_run,
            },
            "data": {
                "n_recipe_embeddings_total": int(len(all_recipe_ids)),
                "n_recipe_embeddings_in_graph": int(in_graph.sum()),
                "embedding_dim": int(all_recipe_vectors_raw.shape[1]),
                "n_unique_sources_in_graph": int(len(set(sources))),
            },
            "centering": centering_diag,
            "raw": raw_metrics,
            "debiased": debiased_metrics,
            "comparison": {
                "delta_n_clusters": int(debiased_metrics["n_clusters"] - raw_metrics["n_clusters"]),
                "delta_modularity_Q": round(float(debiased_metrics["modularity_Q"] - raw_metrics["modularity_Q"]), 6),
                "delta_median_top_source_pct": (
                    round(
                        float(debiased_metrics["median_top_pct"] - raw_metrics["median_top_pct"]),
                        6,
                    )
                    if raw_metrics["median_top_pct"] is not None and debiased_metrics["median_top_pct"] is not None
                    else None
                ),
                "delta_clusters_above_80pct": int(
                    debiased_metrics["n_clusters_above_80pct"] - raw_metrics["n_clusters_above_80pct"]
                ),
                "n_recipes_changed_cluster": int(assignments_df["cluster_changed"].sum()),
                "pct_recipes_changed_cluster": round(
                    100 * float(assignments_df["cluster_changed"].mean()),
                    3,
                ),
            },
            "notes": [
                "Source-centering is a diagnostic mitigation for source/style bias.",
                "Lower source dominance is not automatically better; inspect culinary coherence after debiasing.",
                "Re-run cluster inspection on the debiased embeddings before committing to them.",
            ],
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "raw": raw_metrics,
                "debiased": debiased_metrics,
                "comparison": diagnostics["comparison"],
                "centering": centering_diag,
            }, indent=2, ensure_ascii=False))
            return 0

        write_csv(metrics_df, metrics_path)
        write_csv(cluster_comparison_df, cluster_comparison_path)
        write_csv(source_diag_df, source_diag_path)
        write_csv(assignments_df, assignments_path)
        write_json(diagnostics_path, diagnostics)

        if not args.dry_run:
            recipe_vectors_debiased_full = all_recipe_vectors_raw.copy()
            recipe_vectors_debiased_full[in_graph] = vectors_debiased.astype(np.float32)
            save_debiased_npz(
                arrays,
                recipe_vectors_debiased_full=recipe_vectors_debiased_full,
                out_path=args.embeddings_out,
            )

        log.info("=" * 72)
        log.info("COMPARISON raw -> debiased")
        log.info("  clusters:              %s -> %s", raw_metrics["n_clusters"], debiased_metrics["n_clusters"])
        log.info("  modularity Q:          %.4f -> %.4f", raw_metrics["modularity_Q"], debiased_metrics["modularity_Q"])
        log.info(
            "  median top-source %%:   %s -> %s",
            raw_metrics["median_top_pct"],
            debiased_metrics["median_top_pct"],
        )
        log.info(
            "  clusters >=80%% source: %s -> %s",
            raw_metrics["n_clusters_above_80pct"],
            debiased_metrics["n_clusters_above_80pct"],
        )
        log.info(
            "  recipes changed cluster: %d (%.2f%%)",
            diagnostics["comparison"]["n_recipes_changed_cluster"],
            diagnostics["comparison"]["pct_recipes_changed_cluster"],
        )
        log.info("Done. Next: inspect clusters with the debiased embeddings.")
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
