"""
Label-permutation null model for Cross-Cluster Semantic Coverage (CSC).

The script tests whether the observed CSC components differ from a null model
in which cluster labels are randomly permuted while preserving:

- the citation graph,
- the cluster-size distribution,
- the document corpus, and
- the number of clusters.

For the observed labels and each permutation, it computes:

    S_cross  : cross-cluster citation incidence share
    D_bar    : unweighted mean semantic distance between cluster vocabularies
    D_bar_w  : external-citation-weighted semantic distance
    CSC      : 1 - S_cross * D_bar_w

The semantic topic functions are loaded from an existing ``semantic_a.py``
module, which must expose:

    clean_abstract(text)
    tokenize(text)
    label_topics_c_tfidf(...)

Example
-------
python scripts/null_label_permutation.py \
    --semantic-module src/semantic_a.py \
    --doi-order data/phase0_doi_order.csv \
    --abstracts data/abstracts_full.csv \
    --topics data/paper_topics.csv \
    --edges data/citation_edges_all_crossref.csv \
    --output-dir results/null_label_permutation \
    --permutations 1000 \
    --seed 42

Outputs
-------
null_label_perm_summary.json
    Observed statistics, null means and standard deviations, z-scores, and
    finite-sample permutation p-values.

null_label_perm_dist.npz
    Full null distributions plus the observed statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rbo


LOGGER = logging.getLogger("null_label_permutation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a label-permutation null model for S_cross, D_bar, D_bar_w, "
            "and Cross-Cluster Semantic Coverage (CSC)."
        )
    )
    parser.add_argument(
        "--semantic-module",
        type=Path,
        required=True,
        help="Path to semantic_a.py.",
    )
    parser.add_argument(
        "--doi-order",
        type=Path,
        required=True,
        help="CSV containing the canonical DOI order; requires column 'doi'.",
    )
    parser.add_argument(
        "--abstracts",
        type=Path,
        required=True,
        help="CSV containing columns 'doi' and 'abstract'.",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        required=True,
        help="CSV containing columns 'doi' and 'cluster'.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        required=True,
        help=(
            "Citation-edge CSV containing 'source_doi' and 'target_doi'. "
            "Edges outside the canonical DOI order are ignored."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which result files will be written.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of random label permutations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="NumPy random seed (default: 42).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of ranked terms retained per cluster (default: 50).",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=2,
        help="Minimum n-gram length for c-TF-IDF topics (default: 2).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum n-gram length for c-TF-IDF topics (default: 3).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for c-TF-IDF terms (default: 2).",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="Rank-Biased Overlap persistence parameter (default: 0.9).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N permutations; use 0 to disable.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument ranges and input paths."""
    input_paths = {
        "semantic module": args.semantic_module,
        "DOI order": args.doi_order,
        "abstracts": args.abstracts,
        "topics": args.topics,
        "citation edges": args.edges,
    }
    missing = [f"{name}: {path}" for name, path in input_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n  " + "\n  ".join(missing))

    if args.permutations < 1:
        raise ValueError("--permutations must be at least 1.")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("Require 1 <= --ngram-min <= --ngram-max.")
    if args.min_df < 1:
        raise ValueError("--min-df must be at least 1.")
    if not 0.0 < args.rbo_p < 1.0:
        raise ValueError("--rbo-p must be strictly between 0 and 1.")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative.")


def load_semantic_module(path: Path) -> ModuleType:
    """Load the semantic helper module from a filesystem path."""
    spec = importlib.util.spec_from_file_location("semantic_a", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_functions = (
        "clean_abstract",
        "tokenize",
        "label_topics_c_tfidf",
    )
    missing = [name for name in required_functions if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(
            f"{path} is missing required callable(s): {', '.join(missing)}"
        )
    return module


def require_columns(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    """Raise a clear error when a CSV lacks required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}"
        )


def normalize_doi(series: pd.Series) -> pd.Series:
    """Normalize DOI-like identifiers consistently across input files."""
    return series.astype("string").str.lower().str.strip()


def load_inputs(
    doi_order_path: Path,
    abstracts_path: Path,
    topics_path: Path,
    edges_path: Path,
    semantic_module: ModuleType,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load, align, validate, and preprocess all input data."""
    order = pd.read_csv(doi_order_path)
    require_columns(order, ("doi",), doi_order_path)
    order["doi"] = normalize_doi(order["doi"])

    if order["doi"].isna().any() or (order["doi"] == "").any():
        raise ValueError(f"{doi_order_path} contains missing or empty DOI values.")
    if order["doi"].duplicated().any():
        duplicates = order.loc[order["doi"].duplicated(), "doi"].head(10).tolist()
        raise ValueError(
            f"{doi_order_path} contains duplicate DOI values, including: {duplicates}"
        )

    dois = order["doi"].tolist()
    doi_to_index = {doi: i for i, doi in enumerate(dois)}

    abstracts = pd.read_csv(abstracts_path)
    require_columns(abstracts, ("doi", "abstract"), abstracts_path)
    abstracts["doi"] = normalize_doi(abstracts["doi"])
    abstracts_by_doi = dict(
        zip(abstracts["doi"], abstracts["abstract"].fillna("").astype(str))
    )

    missing_abstracts = sum(doi not in abstracts_by_doi for doi in dois)
    if missing_abstracts:
        LOGGER.warning(
            "%d/%d canonical DOIs have no abstract; empty text will be used.",
            missing_abstracts,
            len(dois),
        )

    processed_abstracts = [
        " ".join(
            semantic_module.tokenize(
                semantic_module.clean_abstract(abstracts_by_doi.get(doi, ""))
            )
        )
        for doi in dois
    ]

    topics = pd.read_csv(topics_path)
    require_columns(topics, ("doi", "cluster"), topics_path)
    topics["doi"] = normalize_doi(topics["doi"])

    if topics["doi"].duplicated().any():
        duplicates = topics.loc[topics["doi"].duplicated(), "doi"].head(10).tolist()
        raise ValueError(
            f"{topics_path} contains duplicate DOI labels, including: {duplicates}"
        )

    labels_by_doi = dict(zip(topics["doi"], topics["cluster"]))
    missing_labels = [doi for doi in dois if doi not in labels_by_doi]
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(
            f"Missing cluster labels for {len(missing_labels)} canonical DOI(s). "
            f"First values: {preview}"
        )

    labels = np.asarray([labels_by_doi[doi] for doi in dois])
    if pd.isna(labels).any():
        raise ValueError(f"{topics_path} contains missing cluster labels.")
    if np.unique(labels).size < 2:
        raise ValueError("At least two distinct cluster labels are required.")

    edges = pd.read_csv(edges_path)
    require_columns(edges, ("source_doi", "target_doi"), edges_path)
    sources = normalize_doi(edges["source_doi"])
    targets = normalize_doi(edges["target_doi"])

    edge_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    skipped_outside = 0
    skipped_self = 0
    skipped_duplicate = 0

    for source_doi, target_doi in zip(sources, targets):
        source_index = doi_to_index.get(source_doi)
        target_index = doi_to_index.get(target_doi)

        if source_index is None or target_index is None:
            skipped_outside += 1
            continue
        if source_index == target_index:
            skipped_self += 1
            continue

        edge = (source_index, target_index)
        if edge in seen:
            skipped_duplicate += 1
            continue

        seen.add(edge)
        edge_pairs.append(edge)

    if not edge_pairs:
        raise ValueError(
            "No usable citation edges remain after DOI alignment, self-loop removal, "
            "and directed-edge deduplication."
        )

    LOGGER.info(
        "Loaded %d documents, %d clusters, and %d directed citation edges.",
        len(dois),
        np.unique(labels).size,
        len(edge_pairs),
    )
    LOGGER.info(
        "Skipped edges: %d outside corpus, %d self-loops, %d duplicates.",
        skipped_outside,
        skipped_self,
        skipped_duplicate,
    )

    edge_source_index = np.fromiter(
        (source for source, _ in edge_pairs), dtype=np.int64, count=len(edge_pairs)
    )
    edge_target_index = np.fromiter(
        (target for _, target in edge_pairs), dtype=np.int64, count=len(edge_pairs)
    )

    return (
        dois,
        processed_abstracts,
        labels,
        edge_source_index,
        edge_target_index,
    )


def compute_components(
    labels: np.ndarray,
    edge_source_index: np.ndarray,
    edge_target_index: np.ndarray,
    processed_abstracts: Sequence[str],
    semantic_module: ModuleType,
    *,
    top_n: int,
    ngram: tuple[int, int],
    min_df: int,
    rbo_p: float,
) -> tuple[float, float, float]:
    """Compute S_cross, D_bar, and D_bar_w for one label assignment."""
    source_labels = labels[edge_source_index]
    target_labels = labels[edge_target_index]

    total_incidence: Counter[Any] = Counter()
    intra_edges: Counter[Any] = Counter()
    external_incidence: Counter[Any] = Counter()

    for source_label, target_label in zip(source_labels, target_labels):
        total_incidence[source_label] += 1

        if source_label != target_label:
            total_incidence[target_label] += 1
            external_incidence[source_label] += 1
            external_incidence[target_label] += 1
        else:
            intra_edges[source_label] += 1

    total_sum = sum(total_incidence.values())
    intra_sum = sum(intra_edges.values())
    if total_sum == 0:
        raise ValueError("Cannot compute S_cross because total citation incidence is zero.")

    s_cross = (total_sum - intra_sum) / total_sum

    ranked_topics = semantic_module.label_topics_c_tfidf(
        processed_abstracts,
        labels,
        top_n=top_n,
        ngram=ngram,
        min_df=min_df,
    )
    clusters = sorted(ranked_topics)
    if len(clusters) < 2:
        return float(s_cross), float("nan"), float("nan")

    distances: dict[tuple[Any, Any], float] = {}
    for left_position, left_cluster in enumerate(clusters):
        for right_cluster in clusters[left_position + 1 :]:
            left_terms = ranked_topics[left_cluster][:top_n]
            right_terms = ranked_topics[right_cluster][:top_n]

            try:
                overlap = rbo.RankingSimilarity(left_terms, right_terms).rbo(p=rbo_p)
            except Exception as exc:  # Preserve the original conservative fallback.
                LOGGER.debug(
                    "RBO failed for cluster pair (%r, %r): %s",
                    left_cluster,
                    right_cluster,
                    exc,
                )
                overlap = 0.0

            distances[(left_cluster, right_cluster)] = 1.0 - float(overlap)

    d_bar = float(np.mean(list(distances.values())))

    weighted_numerator = 0.0
    weighted_denominator = 0.0
    for (left_cluster, right_cluster), distance in distances.items():
        pair_weight = (
            external_incidence.get(left_cluster, 0)
            * external_incidence.get(right_cluster, 0)
        )
        weighted_numerator += pair_weight * distance
        weighted_denominator += pair_weight

    d_bar_w = (
        weighted_numerator / weighted_denominator
        if weighted_denominator > 0
        else float("nan")
    )
    return float(s_cross), d_bar, float(d_bar_w)


def empirical_p_low(observed: float, null: np.ndarray) -> float:
    """Finite-sample lower-tail permutation p-value with +1 correction."""
    return float((1 + np.sum(null <= observed)) / (len(null) + 1))


def empirical_p_high(observed: float, null: np.ndarray) -> float:
    """Finite-sample upper-tail permutation p-value with +1 correction."""
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def z_score(observed: float, null: np.ndarray) -> float:
    """Standardize an observed value against its permutation distribution."""
    standard_deviation = float(np.std(null, ddof=1))
    if standard_deviation == 0.0 or not np.isfinite(standard_deviation):
        return float("nan")
    return float((observed - float(np.mean(null))) / standard_deviation)


def json_number(value: float | int) -> float | int | None:
    """Convert non-finite floats to JSON null for standards-compliant output."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the observed calculation and permutation null model."""
    semantic_module = load_semantic_module(args.semantic_module)
    _, processed_abstracts, real_labels, edge_sources, edge_targets = load_inputs(
        args.doi_order,
        args.abstracts,
        args.topics,
        args.edges,
        semantic_module,
    )

    component_kwargs = {
        "edge_source_index": edge_sources,
        "edge_target_index": edge_targets,
        "processed_abstracts": processed_abstracts,
        "semantic_module": semantic_module,
        "top_n": args.top_n,
        "ngram": (args.ngram_min, args.ngram_max),
        "min_df": args.min_df,
        "rbo_p": args.rbo_p,
    }

    observed_s_cross, observed_d_bar, observed_d_bar_w = compute_components(
        real_labels,
        **component_kwargs,
    )
    observed_csc = 1.0 - observed_s_cross * observed_d_bar_w

    LOGGER.info(
        "Observed | S_cross=%.4f D_bar=%.4f D_bar_w=%.4f CSC=%.4f",
        observed_s_cross,
        observed_d_bar,
        observed_d_bar_w,
        observed_csc,
    )

    rng = np.random.default_rng(args.seed)
    null_s_cross = np.empty(args.permutations, dtype=float)
    null_d_bar = np.empty(args.permutations, dtype=float)
    null_d_bar_w = np.empty(args.permutations, dtype=float)
    null_csc = np.empty(args.permutations, dtype=float)

    started = time.perf_counter()
    for permutation_index in range(args.permutations):
        permuted_labels = rng.permutation(real_labels)
        s_cross, d_bar, d_bar_w = compute_components(
            permuted_labels,
            **component_kwargs,
        )

        null_s_cross[permutation_index] = s_cross
        null_d_bar[permutation_index] = d_bar
        null_d_bar_w[permutation_index] = d_bar_w
        null_csc[permutation_index] = 1.0 - s_cross * d_bar_w

        completed = permutation_index + 1
        if args.progress_every and completed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "Permutation progress: %d/%d | elapsed=%.1fs",
                completed,
                args.permutations,
                elapsed,
            )

    cluster_count = int(np.unique(real_labels).size)
    result: dict[str, Any] = {
        "B": args.permutations,
        "k": cluster_count,
        "seed": args.seed,
        "parameters": {
            "top_n": args.top_n,
            "ngram": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "rbo_p": args.rbo_p,
        },
        "observed": {
            "s_cross": observed_s_cross,
            "d_bar": observed_d_bar,
            "d_bar_w": observed_d_bar_w,
            "csc": observed_csc,
        },
        "null_mean": {
            "s_cross": float(np.mean(null_s_cross)),
            "d_bar": float(np.mean(null_d_bar)),
            "d_bar_w": float(np.mean(null_d_bar_w)),
            "csc": float(np.mean(null_csc)),
        },
        "null_sd": {
            "s_cross": float(np.std(null_s_cross, ddof=1)),
            "d_bar": float(np.std(null_d_bar, ddof=1)),
            "d_bar_w": float(np.std(null_d_bar_w, ddof=1)),
            "csc": float(np.std(null_csc, ddof=1)),
        },
        "z": {
            "s_cross": z_score(observed_s_cross, null_s_cross),
            "d_bar": z_score(observed_d_bar, null_d_bar),
            "d_bar_w": z_score(observed_d_bar_w, null_d_bar_w),
            "csc": z_score(observed_csc, null_csc),
        },
        "p_csc_low": empirical_p_low(observed_csc, null_csc),
        "p_dbarw_high": empirical_p_high(observed_d_bar_w, null_d_bar_w),
        "p_scross_low": empirical_p_low(observed_s_cross, null_s_cross),
        "p_scross_high": empirical_p_high(observed_s_cross, null_s_cross),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "null_label_perm_summary.json"
    distribution_path = args.output_dir / "null_label_perm_dist.npz"

    json_ready = {
        key: (
            {nested_key: json_number(nested_value) for nested_key, nested_value in value.items()}
            if isinstance(value, dict)
            else json_number(value)
        )
        for key, value in result.items()
    }
    # Parameters contains a list and needs no numeric conversion beyond standard types.
    json_ready["parameters"] = result["parameters"]

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    np.savez_compressed(
        distribution_path,
        s_cross=null_s_cross,
        d_bar=null_d_bar,
        d_bar_w=null_d_bar_w,
        csc=null_csc,
        obs=np.array(
            [
                observed_s_cross,
                observed_d_bar,
                observed_d_bar_w,
                observed_csc,
            ],
            dtype=float,
        ),
    )

    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", distribution_path)
    LOGGER.info("Done | %s", json.dumps(json_ready, ensure_ascii=False))
    return result


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
