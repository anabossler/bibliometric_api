"""
step3_0c_flag_quality.py
========================

Flag recipes with clustering-quality issues in the canonical RELISH graph.

This step re-runs the chosen k-NN + Leiden clustering and writes a transparent
quality flag to Recipe nodes:

  - high
      Recipe is clustered into a culinarily coherent group by default.

  - ner_residual_noise
      Recipe belongs to a cluster whose top distinctive ingredients contain
      many structural-junk tokens, suggesting the cluster is held together by
      NER residue rather than culinary signal.

  - untranslated_corpus
      Recipe belongs to a known source where translation/normalization left
      raw foreign-language or unit tokens that should be interpreted cautiously.

  - not_clustered_missing_embedding
      Recipe exists in the graph but has no recipe embedding and therefore was
      not assigned to a cluster.

The graph is not changed destructively: this adds node attributes and writes a
full audit trail so the flagging is transparent and reversible.

Default inputs
--------------

  data/graph_step2_canonical.gpickle
  data/layer_2_embeddings.npz

Default outputs
---------------

  data/graph_step3_flagged.gpickle
  data/clustering_quality_flags.csv
  data/clustering_quality_cluster_diagnostics.csv
  data/clustering_quality_stats.json

Usage
-----

  python step3_0c_flag_quality.py

  python step3_0c_flag_quality.py --k 30 --junk-threshold 5

  python step3_0c_flag_quality.py --dry-run

  python step3_0c_flag_quality.py \
    --untranslated-source historische_esskultur="German modern cookbook, untranslated ingredients"

Dependencies
------------

  pip install numpy pandas scikit-learn igraph leidenalg

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GRAPH_IN = Path("data/graph_step2_canonical.gpickle")
DEFAULT_GRAPH_OUT = Path("data/graph_step3_flagged.gpickle")
DEFAULT_EMBEDDINGS = Path("data/layer_2_embeddings.npz")
DEFAULT_DATA_DIR = Path("data")

DEFAULT_RANDOM_STATE = 42
DEFAULT_K = 30
DEFAULT_JUNK_THRESHOLD = 5
DEFAULT_TOP_N_DISTINCTIVE = 15
DEFAULT_MIN_CLUSTER_INGREDIENT_PCT = 0.05
DEFAULT_DISTINCTIVENESS_SMOOTHING = 0.001

DEFAULT_UNTRANSLATED_SOURCES = {
    "historische_esskultur": "German modern cookbook, untranslated ingredients",
}

DEFAULT_UNIT_TOKENS = {
    "kg", "g", "gr", "mg", "ml", "l", "tl", "el", "tsp", "tbsp", "oz", "lb",
    "cup", "cups", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
}

DEFAULT_GERMAN_LEAKAGE = {
    "zucker", "rosinen", "eigelb", "elstar", "vanilleschote",
    "puderzucker", "schmelzen", "milch", "wasser", "essig",
    "schmand", "speck", "salz", "mehl", "butter", "buttermilch",
    "zitrone", "apfel", "birne", "kirschen", "pflaumen",
    "kraphen", "strudelteig", "knodel", "knoedel", "spatzle", "spaetzle",
    "lauwarme", "lauwarm", "geschmolzen", "gleichm", "langsam",
    "aufrollen", "goldbraun", "zimt", "tl_zimt",
}

DEFAULT_META_TEXT_TOKENS = {
    "recipe", "recipes", "english", "medieval", "modern", "spelling",
    "collection", "time", "gastronomic", "culture", "country",
    "privada", "catal", "translated", "translation", "ingredient",
    "ingredients",
}

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step3_0c_flag_quality")


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


def load_data(
    *,
    graph_path: Path,
    embeddings_path: Path,
    allow_pickle_npz: bool,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, list[str]]]:
    graph = load_graph(graph_path)
    all_ids, all_vectors = load_recipe_embeddings(embeddings_path, allow_pickle=allow_pickle_npz)

    in_graph = np.array([str(node_id) in graph for node_id in all_ids], dtype=bool)
    ids = all_ids[in_graph]
    vectors = all_vectors[in_graph]

    if len(ids) == 0:
        raise ValueError("No recipe embeddings match nodes in the graph")

    vectors = normalize_vectors(vectors)

    recipe_to_ingredients: dict[str, list[str]] = defaultdict(list)

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "contains":
            continue
        if graph.nodes.get(source, {}).get("node_type") != "Recipe":
            continue
        if graph.nodes.get(target, {}).get("node_type") != "Ingredient":
            continue

        recipe_to_ingredients[str(source)].append(str(target))

    recipe_to_ingredients = {
        recipe_id: sorted(set(ingredients))
        for recipe_id, ingredients in recipe_to_ingredients.items()
    }

    log.info("Recipes with embeddings in graph: %d", len(ids))
    log.info("Ingredient lists built for %d recipes", len(recipe_to_ingredients))

    return graph, vectors, ids, recipe_to_ingredients


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def build_knn_graph(vectors: np.ndarray, *, k: int):
    """Build weighted undirected igraph k-NN graph.

    Mutual k-NN edges are collapsed by max similarity so weights remain in
    [0, 1], consistent with step3_0_explore_k.py and step3_0b_inspect_clusters.py.
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
# Junk detection
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def ingredient_name(token: str) -> str:
    return str(token).replace("ing::", "")


def normalized_parts(token: str) -> list[str]:
    name = strip_accents(ingredient_name(token)).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return [part for part in name.split("_") if part]


def is_junk_token(
    token: str,
    *,
    unit_tokens: set[str],
    untranslated_tokens: set[str],
    meta_text_tokens: set[str],
    phrase_word_threshold: int,
) -> tuple[bool, str]:
    """Return whether an ingredient token looks like structural NER junk."""
    parts = normalized_parts(token)

    if not parts:
        return True, "empty_token"

    if len(parts) >= phrase_word_threshold:
        return True, f"phrase_fragment>={phrase_word_threshold}_parts"

    if any(part in unit_tokens for part in parts):
        return True, "unit_or_measurement_token"

    if any(part in untranslated_tokens for part in parts):
        return True, "known_untranslated_token"

    if any(part in meta_text_tokens for part in parts):
        return True, "meta_text_token"

    if any(re.fullmatch(r"\d+\w*", part) for part in parts):
        return True, "numeric_token"

    return False, ""


def distinctive_ingredient_scores(
    *,
    cluster_recipe_ids: list[str],
    other_recipe_ids: list[str],
    recipe_to_ingredients: dict[str, list[str]],
    min_cluster_pct: float,
    smoothing: float,
) -> list[dict[str, Any]]:
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
        score = cluster_pct / (other_pct + smoothing)

        rows.append({
            "ingredient_id": ingredient_id,
            "ingredient": ingredient_name(ingredient_id),
            "score": float(score),
            "cluster_count": int(cluster_count),
            "other_count": int(other_count),
            "cluster_pct": float(cluster_pct),
            "other_pct": float(other_pct),
        })

    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["cluster_pct"],
            -row["cluster_count"],
            row["ingredient"],
        )
    )

    return rows


def diagnose_clusters(
    *,
    labels: np.ndarray,
    ids: np.ndarray,
    recipe_to_ingredients: dict[str, list[str]],
    top_n_distinctive: int,
    min_cluster_pct: float,
    smoothing: float,
    junk_threshold: int,
    unit_tokens: set[str],
    untranslated_tokens: set[str],
    meta_text_tokens: set[str],
    phrase_word_threshold: int,
) -> tuple[set[int], pd.DataFrame]:
    noisy_clusters: set[int] = set()
    all_indices = np.arange(len(ids))
    rows: list[dict[str, Any]] = []

    log.info(
        "Detecting NER-noisy clusters: >=%d junk tokens in top %d distinctive ingredients",
        junk_threshold,
        top_n_distinctive,
    )

    for cluster_id in sorted(set(labels)):
        cluster_indices = all_indices[labels == cluster_id]
        other_indices = all_indices[labels != cluster_id]

        cluster_recipe_ids = [str(ids[index]) for index in cluster_indices]
        other_recipe_ids = [str(ids[index]) for index in other_indices]

        distinctive = distinctive_ingredient_scores(
            cluster_recipe_ids=cluster_recipe_ids,
            other_recipe_ids=other_recipe_ids,
            recipe_to_ingredients=recipe_to_ingredients,
            min_cluster_pct=min_cluster_pct,
            smoothing=smoothing,
        )[:top_n_distinctive]

        junk_items: list[dict[str, str]] = []
        for item in distinctive:
            is_junk, reason = is_junk_token(
                item["ingredient_id"],
                unit_tokens=unit_tokens,
                untranslated_tokens=untranslated_tokens,
                meta_text_tokens=meta_text_tokens,
                phrase_word_threshold=phrase_word_threshold,
            )
            if is_junk:
                junk_items.append({
                    "ingredient_id": item["ingredient_id"],
                    "ingredient": item["ingredient"],
                    "reason": reason,
                })

        junk_count = len(junk_items)
        is_noisy = junk_count >= junk_threshold

        if is_noisy:
            noisy_clusters.add(int(cluster_id))
            log.info(
                "Cluster %d (n=%d) flagged: %d junk tokens; examples=%s",
                cluster_id,
                len(cluster_indices),
                junk_count,
                [item["ingredient"] for item in junk_items[:5]],
            )
        else:
            log.info(
                "Cluster %d (n=%d) clean: %d junk in top %d",
                cluster_id,
                len(cluster_indices),
                junk_count,
                top_n_distinctive,
            )

        rows.append({
            "cluster": int(cluster_id),
            "n": int(len(cluster_indices)),
            "junk_count": int(junk_count),
            "junk_threshold": int(junk_threshold),
            "is_noisy": bool(is_noisy),
            "junk_examples": "; ".join(
                f"{item['ingredient']}:{item['reason']}" for item in junk_items[:10]
            ),
            "top_distinctive_ingredients": "; ".join(
                f"{item['ingredient']}({item['score']:.2f})" for item in distinctive
            ),
        })

    diagnostics = pd.DataFrame(rows).sort_values(
        by=["is_noisy", "junk_count", "n"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    if not noisy_clusters:
        log.warning("No noisy clusters detected. The threshold may be too high.")

    return noisy_clusters, diagnostics


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def parse_untranslated_sources(items: list[str], path: Path | None) -> dict[str, str]:
    sources = dict(DEFAULT_UNTRANSLATED_SOURCES)

    def add_item(raw: str) -> None:
        if not raw.strip():
            return
        if "=" in raw:
            source, _, reason = raw.partition("=")
            sources[source.strip()] = reason.strip() or "user-specified untranslated source"
        else:
            sources[raw.strip()] = "user-specified untranslated source"

    for item in items or []:
        add_item(item)

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Untranslated-source file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                add_item(line)

    return sources


def load_extra_tokens(items: list[str], path: Path | None) -> set[str]:
    tokens = {token.strip().lower() for token in items or [] if token.strip()}

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")
        tokens.update(
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    return tokens


def assign_flags(
    *,
    graph: Any,
    ids: np.ndarray,
    labels: np.ndarray,
    noisy_clusters: set[int],
    untranslated_sources: dict[str, str],
    missing_embedding_flag: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    id_to_cluster = {str(ids[index]): int(labels[index]) for index in range(len(ids))}
    embedded_recipe_ids = set(id_to_cluster)

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        recipe_id = str(node_id)
        source_id = attrs.get("source_id")
        cluster_id = id_to_cluster.get(recipe_id)

        # Priority: known untranslated corpus > noisy cluster > missing embedding > high.
        if source_id in untranslated_sources:
            flag = "untranslated_corpus"
            reason = f"source_id={source_id}: {untranslated_sources[source_id]}"
        elif cluster_id in noisy_clusters:
            flag = "ner_residual_noise"
            reason = f"recipe in cluster {cluster_id} with anomalous distinctive-ingredient signature"
        elif recipe_id not in embedded_recipe_ids:
            flag = missing_embedding_flag
            reason = "Recipe node has no matching recipe embedding and was not clustered"
        else:
            flag = "high"
            reason = ""

        counts[flag] += 1

        rows.append({
            "recipe_id": recipe_id,
            "title": attrs.get("title"),
            "source_id": source_id,
            "source_language": attrs.get("source_language"),
            "source_place": attrs.get("source_place"),
            "period_derived": attrs.get("period_derived"),
            "cluster_id": cluster_id,
            "clustering_quality_flag": flag,
            "reason": reason,
        })

    return rows, counts


def apply_flags_to_graph(graph: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        recipe_id = row["recipe_id"]
        if recipe_id not in graph:
            continue
        graph.nodes[recipe_id]["clustering_quality_flag"] = row["clustering_quality_flag"]
        graph.nodes[recipe_id]["clustering_quality_reason"] = row["reason"]
        graph.nodes[recipe_id]["cluster_id"] = row["cluster_id"]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_graph(graph: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote flagged graph: %s", path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--graph-in",
        type=Path,
        default=DEFAULT_GRAPH_IN,
        help=f"Input canonical graph. Default: {DEFAULT_GRAPH_IN}",
    )
    parser.add_argument(
        "--graph-out",
        type=Path,
        default=DEFAULT_GRAPH_OUT,
        help=f"Output flagged graph. Default: {DEFAULT_GRAPH_OUT}",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Layer 2 embeddings NPZ. Default: {DEFAULT_EMBEDDINGS}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="k_NN value. Should match the chosen clustering.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Optional Leiden RBConfiguration resolution. Omit for modularity partition.",
    )
    parser.add_argument(
        "--junk-threshold",
        type=int,
        default=DEFAULT_JUNK_THRESHOLD,
        help="Flag cluster as ner_residual_noise if at least this many top distinctive ingredients are junk.",
    )
    parser.add_argument(
        "--top-n-distinctive",
        type=int,
        default=DEFAULT_TOP_N_DISTINCTIVE,
        help="Number of top distinctive ingredients inspected per cluster.",
    )
    parser.add_argument(
        "--min-cluster-ingredient-pct",
        type=float,
        default=DEFAULT_MIN_CLUSTER_INGREDIENT_PCT,
        help="Minimum ingredient prevalence inside a cluster to inspect.",
    )
    parser.add_argument(
        "--distinctiveness-smoothing",
        type=float,
        default=DEFAULT_DISTINCTIVENESS_SMOOTHING,
        help="Smoothing added to outside-cluster prevalence.",
    )
    parser.add_argument(
        "--phrase-word-threshold",
        type=int,
        default=4,
        help="Ingredient names with at least this many underscore-separated parts are considered phrase fragments.",
    )
    parser.add_argument(
        "--untranslated-source",
        action="append",
        default=[],
        help=(
            "Known untranslated source as source_id or source_id=reason. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--untranslated-source-file",
        type=Path,
        help="Optional file with one source_id or source_id=reason per line.",
    )
    parser.add_argument(
        "--junk-token",
        action="append",
        default=[],
        help="Additional untranslated/junk token. Can be repeated.",
    )
    parser.add_argument(
        "--junk-token-file",
        type=Path,
        help="Optional file with one additional untranslated/junk token per line.",
    )
    parser.add_argument(
        "--unit-token",
        action="append",
        default=[],
        help="Additional unit/measurement token. Can be repeated.",
    )
    parser.add_argument(
        "--meta-text-token",
        action="append",
        default=[],
        help="Additional meta-text token. Can be repeated.",
    )
    parser.add_argument(
        "--missing-embedding-flag",
        default="not_clustered_missing_embedding",
        help="Flag assigned to Recipe nodes without recipe embeddings.",
    )
    parser.add_argument(
        "--flags-out",
        type=Path,
        help="Path to per-recipe flags CSV.",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        help="Path to cluster diagnostics CSV.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="Path to stats JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be flagged but do not write graph.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write CSV/JSON audit outputs.",
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

    args.flags_out = args.flags_out or args.data_dir / "clustering_quality_flags.csv"
    args.diagnostics_out = args.diagnostics_out or args.data_dir / "clustering_quality_cluster_diagnostics.csv"
    args.stats_out = args.stats_out or args.data_dir / "clustering_quality_stats.json"

    return args


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
    if args.junk_threshold < 1:
        log.error("--junk-threshold must be >= 1")
        return 2
    if args.top_n_distinctive < 1:
        log.error("--top-n-distinctive must be >= 1")
        return 2
    if not (0 <= args.min_cluster_ingredient_pct <= 1):
        log.error("--min-cluster-ingredient-pct must be in [0, 1]")
        return 2
    if args.distinctiveness_smoothing < 0:
        log.error("--distinctiveness-smoothing must be >= 0")
        return 2
    if args.phrase_word_threshold < 2:
        log.error("--phrase-word-threshold must be >= 2")
        return 2

    try:
        graph, vectors, ids, recipe_to_ingredients = load_data(
            graph_path=args.graph_in,
            embeddings_path=args.embeddings,
            allow_pickle_npz=args.allow_pickle_npz,
        )

        knn_graph, k_effective = build_knn_graph(vectors, k=args.k)
        labels, modularity = run_leiden(
            knn_graph,
            random_state=args.random_state,
            resolution=args.resolution,
        )

        untranslated_sources = parse_untranslated_sources(
            args.untranslated_source,
            args.untranslated_source_file,
        )

        unit_tokens = set(DEFAULT_UNIT_TOKENS) | load_extra_tokens(args.unit_token, None)
        untranslated_tokens = (
            set(DEFAULT_GERMAN_LEAKAGE)
            | load_extra_tokens(args.junk_token, args.junk_token_file)
        )
        meta_text_tokens = set(DEFAULT_META_TEXT_TOKENS) | load_extra_tokens(args.meta_text_token, None)

        noisy_clusters, diagnostics_df = diagnose_clusters(
            labels=labels,
            ids=ids,
            recipe_to_ingredients=recipe_to_ingredients,
            top_n_distinctive=args.top_n_distinctive,
            min_cluster_pct=args.min_cluster_ingredient_pct,
            smoothing=args.distinctiveness_smoothing,
            junk_threshold=args.junk_threshold,
            unit_tokens=unit_tokens,
            untranslated_tokens=untranslated_tokens,
            meta_text_tokens=meta_text_tokens,
            phrase_word_threshold=args.phrase_word_threshold,
        )

        flag_rows, flag_counts = assign_flags(
            graph=graph,
            ids=ids,
            labels=labels,
            noisy_clusters=noisy_clusters,
            untranslated_sources=untranslated_sources,
            missing_embedding_flag=args.missing_embedding_flag,
        )

        flags_df = pd.DataFrame(flag_rows).sort_values(
            by=["clustering_quality_flag", "source_id", "recipe_id"],
            kind="mergesort",
        )

        log.info("Flag distribution: %s", dict(flag_counts))

        stats = {
            "inputs": {
                "graph_in": str(args.graph_in),
                "embeddings": str(args.embeddings),
            },
            "outputs": {
                "graph_out": None if args.dry_run else str(args.graph_out),
                "flags_csv": str(args.flags_out),
                "cluster_diagnostics_csv": str(args.diagnostics_out),
                "stats_json": str(args.stats_out),
            },
            "parameters": {
                "k": args.k,
                "k_effective": k_effective,
                "resolution": args.resolution,
                "junk_threshold": args.junk_threshold,
                "top_n_distinctive": args.top_n_distinctive,
                "min_cluster_ingredient_pct": args.min_cluster_ingredient_pct,
                "distinctiveness_smoothing": args.distinctiveness_smoothing,
                "phrase_word_threshold": args.phrase_word_threshold,
                "random_state": args.random_state,
                "allow_pickle_npz": args.allow_pickle_npz,
                "dry_run": args.dry_run,
            },
            "clustering": {
                "n_embedded_recipes": int(len(ids)),
                "n_clusters": int(len(set(labels))),
                "modularity_Q": round(float(modularity), 6),
                "n_edges": int(knn_graph.ecount()),
            },
            "flag_counts": {str(key): int(value) for key, value in flag_counts.items()},
            "noisy_clusters": sorted(int(cluster) for cluster in noisy_clusters),
            "untranslated_sources": untranslated_sources,
            "total_recipe_nodes": int(len(flag_rows)),
            "pct_high": round(100 * flag_counts["high"] / max(1, len(flag_rows)), 2),
            "pct_flagged": round(
                100 * (1 - flag_counts["high"] / max(1, len(flag_rows))),
                2,
            ),
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "flag_counts": stats["flag_counts"],
                "noisy_clusters": stats["noisy_clusters"],
                "pct_high": stats["pct_high"],
                "pct_flagged": stats["pct_flagged"],
            }, indent=2, ensure_ascii=False))
            return 0

        args.data_dir.mkdir(parents=True, exist_ok=True)

        write_csv(flags_df, args.flags_out)
        write_csv(diagnostics_df, args.diagnostics_out)
        write_json(args.stats_out, stats)

        if not args.dry_run:
            apply_flags_to_graph(graph, flag_rows)
            write_graph(graph, args.graph_out)

        log.info("Done.")
        log.info("%.1f%% of Recipe nodes flagged high", stats["pct_high"])
        log.info("%.1f%% flagged for known limitations", stats["pct_flagged"])
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
