"""
=====================================

Detect ingredient canonicalization candidates from Step 2 embeddings.


Rationale
---------

Many Ingredient nodes are NER-induced variants of the same concept, for example:

  wine / beverage_wine / grape_wine / wormwood_wine

Embedding similarity can expose these variants. However, expert validation is
required to avoid:

  1. collapsing functionally distinct ingredients
     e.g. almond_seed vs almond_milk

  2. destroying cross-lingual fragmentation
     e.g. salt vs salz, which may be analytically important for AWS-style
     vocabulary-fragmentation analysis

Reads
-----

  data/layer_2_embeddings.npz
  data/graph_step1.gpickle

Writes
------

  data/ingredient_canonicalization_proposals.csv
  data/ingredient_similarity_histogram.txt
  data/ingredient_canonicalization_stats.json

Usage
-----

  python step2_5_canonicalization_proposals.py

  python step2_5_canonicalization_proposals.py --min-sim 0.85

  python step2_5_canonicalization_proposals.py \
    --embeddings data/layer_2_embeddings.npz \
    --graph data/graph_step1.gpickle \
    --output-dir data/

  python step2_5_canonicalization_proposals.py --max-proposals 5000

Output CSV
----------

The CSV includes an `accepted` column, initially empty. A human reviewer should
fill it with TRUE/FALSE before any merge-application step is run.


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step1.gpickle")
DEFAULT_EMBEDDINGS = Path("data/layer_2_embeddings.npz")
DEFAULT_OUTPUT_DIR = Path("data")

DEFAULT_MIN_SIM = 0.85
AUTO_SAFE_SIM = 0.97
AUTO_SAFE_WITH_SUBSTRING = 0.95
REVIEW_FLOOR_WITH_SUBSTRING = 0.90
DEFAULT_CHUNK_SIZE = 256

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step2_5_canonicalization_proposals")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngredientMeta:
    node_id: str
    label: str
    n_occurrences: int
    ner_noise_flag: bool


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_prefix(node_id: str) -> str:
    """Return 'wine' from 'ing::wine'."""
    return node_id.split("::", 1)[1] if "::" in node_id else node_id


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_name(text: str) -> str:
    text = strip_prefix(str(text))
    text = strip_accents(text).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return {
        token
        for token in normalize_name(text).split()
        if len(token) >= 3
    }


def share_substring(a: str, b: str, min_len: int = 3) -> bool:
    """Return True if names share a contiguous substring of at least min_len."""
    a_norm = normalize_name(a).replace(" ", "")
    b_norm = normalize_name(b).replace(" ", "")

    if len(a_norm) < min_len or len(b_norm) < min_len:
        return False

    short, long = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)

    for index in range(len(short) - min_len + 1):
        if short[index:index + min_len] in long:
            return True

    return False


def token_overlap(a: str, b: str) -> bool:
    return bool(token_set(a) & token_set(b))


def containment_relation(a: str, b: str) -> bool:
    """Return True for cases like 'red wine' vs 'wine'."""
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return False

    return a_norm in b_norm or b_norm in a_norm


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_action(
    sim: float,
    *,
    shares_substring: bool,
    shares_token: bool,
    containment: bool,
    exact_normalized_match: bool,
) -> tuple[str, str, int]:
    """Return (suggested_action, reasoning, review_priority).

    `auto_safe` is still only a proposal. It does not mean "merge without human
    validation"; it means "review first because this is likely a harmless
    spelling/NER variant."

    To protect the AWS cross-lingual-fragmentation signal, very high similarity
    without lexical overlap remains `review`, not `auto_safe`.
    """
    if exact_normalized_match:
        return "auto_safe", "exact normalized name match", 1

    if sim >= AUTO_SAFE_SIM and (shares_substring or shares_token or containment):
        return "auto_safe", f"sim>={AUTO_SAFE_SIM:.2f} with lexical overlap", 1

    if sim >= AUTO_SAFE_WITH_SUBSTRING and (shares_substring or shares_token or containment):
        return "auto_safe", f"sim>={AUTO_SAFE_WITH_SUBSTRING:.2f} with lexical overlap", 2

    if sim >= AUTO_SAFE_SIM:
        return "review", "very high sim but no lexical overlap; possible cross-lingual or functional distinction", 1

    if sim >= REVIEW_FLOOR_WITH_SUBSTRING and (shares_substring or shares_token or containment):
        return "review", "moderate/high sim with lexical overlap", 2

    return "review", "moderate sim without lexical overlap; likely keep separate unless expert confirms", 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(path: Path, *, allow_pickle: bool) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    log.info("Loading embeddings: %s", path)
    with np.load(path, allow_pickle=allow_pickle) as npz:
        required = {"ingredients_node_ids", "ingredients_vectors"}
        missing = required - set(npz.files)
        if missing:
            raise ValueError(f"Embeddings NPZ is missing required arrays: {sorted(missing)}")

        ids = npz["ingredients_node_ids"].astype(str)
        vectors = np.asarray(npz["ingredients_vectors"], dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"ingredients_vectors must be 2D, got shape {vectors.shape}")

    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"ingredients_node_ids length {len(ids)} does not match "
            f"ingredients_vectors rows {vectors.shape[0]}"
        )

    if len(ids) == 0:
        raise ValueError("No ingredient embeddings found")

    log.info("Ingredient embeddings: %d nodes, dim=%d", vectors.shape[0], vectors.shape[1])
    return ids, vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or Inf values")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    n_zero = int((norms[:, 0] < 1e-12).sum())
    if n_zero:
        raise ValueError(f"Embedding matrix contains {n_zero} near-zero vectors")

    return vectors / np.maximum(norms, 1e-12)


def load_graph_metadata(path: Path) -> dict[str, IngredientMeta]:
    """Load ingredient occurrence counts and noise flags from trusted graph."""
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph metadata: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    metadata: dict[str, IngredientMeta] = {}

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Ingredient":
            continue

        node_id_str = str(node_id)
        label = (
            attrs.get("canonical_name")
            or attrs.get("name")
            or attrs.get("label")
            or strip_prefix(node_id_str)
        )

        try:
            n_occurrences = int(attrs.get("n_occurrences", 0) or 0)
        except (TypeError, ValueError):
            n_occurrences = 0

        metadata[node_id_str] = IngredientMeta(
            node_id=node_id_str,
            label=str(label),
            n_occurrences=n_occurrences,
            ner_noise_flag=bool(attrs.get("ner_noise_flag", False)),
        )

    log.info("Loaded metadata for %d ingredient nodes", len(metadata))
    return metadata


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def similarity_bucket_index(values: np.ndarray, n_buckets: int) -> np.ndarray:
    clipped = np.clip(values, 0.0, np.nextafter(1.0, 0.0))
    return np.floor(clipped * n_buckets).astype(np.int32)


def choose_root(
    id_a: str,
    id_b: str,
    meta: dict[str, IngredientMeta],
) -> tuple[str, str, IngredientMeta, IngredientMeta]:
    """Choose canonical root as the more frequent ingredient, deterministic tie."""
    fallback_a = IngredientMeta(id_a, strip_prefix(id_a), 0, False)
    fallback_b = IngredientMeta(id_b, strip_prefix(id_b), 0, False)

    meta_a = meta.get(id_a, fallback_a)
    meta_b = meta.get(id_b, fallback_b)

    if (meta_a.n_occurrences, id_b) >= (meta_b.n_occurrences, id_a):
        return id_a, id_b, meta_a, meta_b

    return id_b, id_a, meta_b, meta_a


def proposal_row(
    *,
    id_a: str,
    id_b: str,
    sim: float,
    meta: dict[str, IngredientMeta],
) -> dict[str, Any]:
    root_id, merged_id, root_meta, merged_meta = choose_root(id_a, id_b, meta)

    root_norm = normalize_name(root_meta.label or root_id)
    merged_norm = normalize_name(merged_meta.label or merged_id)

    shares_sub = share_substring(root_norm, merged_norm)
    shares_tok = token_overlap(root_norm, merged_norm)
    containment = containment_relation(root_norm, merged_norm)
    exact_match = bool(root_norm and root_norm == merged_norm)

    action, reasoning, priority = classify_action(
        sim,
        shares_substring=shares_sub,
        shares_token=shares_tok,
        containment=containment,
        exact_normalized_match=exact_match,
    )

    return {
        "canonical_root_id": root_id,
        "canonical_root_label": root_meta.label,
        "merged_id": merged_id,
        "merged_label": merged_meta.label,
        "cosine_sim": round(float(sim), 6),
        "n_occurrences_root": int(root_meta.n_occurrences),
        "n_occurrences_merged": int(merged_meta.n_occurrences),
        "shares_substring": bool(shares_sub),
        "shares_token": bool(shares_tok),
        "name_containment": bool(containment),
        "exact_normalized_match": bool(exact_match),
        "root_ner_noise_flag": bool(root_meta.ner_noise_flag),
        "merged_ner_noise_flag": bool(merged_meta.ner_noise_flag),
        "suggested_action": action,
        "review_priority": int(priority),
        "reasoning": reasoning,
        "accepted": "",
    }


def compute_pair_proposals(
    vectors: np.ndarray,
    ids: np.ndarray,
    metadata: dict[str, IngredientMeta],
    *,
    min_sim: float,
    chunk_size: int,
    histogram_buckets: int,
    max_proposals: int | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Compute all ingredient pairs with cosine similarity >= min_sim.

    Returns:
      proposals_df, histogram_buckets, stats
    """
    n = vectors.shape[0]
    histogram = np.zeros(histogram_buckets, dtype=np.int64)
    rows: list[dict[str, Any]] = []

    log.info("Scanning %d x %d similarity space in chunks of %d", n, n, chunk_size)

    total_pairs = n * (n - 1) // 2
    n_above_threshold = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = vectors[start:end] @ vectors.T

        for local_i in range(end - start):
            global_i = start + local_i
            row = block[local_i, global_i + 1:]
            if row.size == 0:
                continue

            bucket_idx = similarity_bucket_index(row, histogram_buckets)
            np.add.at(histogram, bucket_idx, 1)

            matches = np.where(row >= min_sim)[0]
            if len(matches) == 0:
                continue

            n_above_threshold += int(len(matches))

            for offset in matches:
                j = global_i + 1 + int(offset)
                row_dict = proposal_row(
                    id_a=str(ids[global_i]),
                    id_b=str(ids[j]),
                    sim=float(row[offset]),
                    meta=metadata,
                )
                rows.append(row_dict)

                if max_proposals is not None and len(rows) >= max_proposals:
                    log.warning(
                        "Reached --max-proposals=%d; stopping proposal collection early",
                        max_proposals,
                    )
                    break

            if max_proposals is not None and len(rows) >= max_proposals:
                break

        if max_proposals is not None and len(rows) >= max_proposals:
            break

        if start == 0 or (start // chunk_size) % 4 == 0:
            log.info("Rows scanned: %d/%d, proposals so far: %d", end, n, len(rows))

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=[
                "review_priority",
                "suggested_action",
                "cosine_sim",
                "n_occurrences_root",
                "canonical_root_id",
                "merged_id",
            ],
            ascending=[True, True, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    counts = {
        "n_ingredients": int(n),
        "total_pairs_upper_triangle": int(total_pairs),
        "pairs_above_min_sim": int(n_above_threshold),
        "proposals_written": int(len(df)),
        "max_proposals_reached": bool(max_proposals is not None and len(df) >= max_proposals),
        "auto_safe": int((df["suggested_action"] == "auto_safe").sum()) if not df.empty else 0,
        "review": int((df["suggested_action"] == "review").sum()) if not df.empty else 0,
        "involving_ner_noise": int(
            (df["root_ner_noise_flag"] | df["merged_ner_noise_flag"]).sum()
        ) if not df.empty else 0,
        "exact_normalized_match": int(df["exact_normalized_match"].sum()) if not df.empty else 0,
        "with_token_overlap": int(df["shares_token"].sum()) if not df.empty else 0,
        "with_substring_overlap": int(df["shares_substring"].sum()) if not df.empty else 0,
    }

    return df, histogram, counts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_histogram(histogram: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total = int(histogram.sum())
    if total == 0:
        path.write_text("(no pairs)\n", encoding="utf-8")
        log.info("Wrote histogram: %s", path)
        return

    max_count = int(histogram.max())
    bar_width = 60
    n_buckets = len(histogram)

    lines = [
        "Pairwise cosine similarity histogram (Ingredient x Ingredient)",
        "",
    ]

    for index, count in enumerate(histogram):
        low = index / n_buckets
        high = (index + 1) / n_buckets
        bar = "#" * int(bar_width * count / max_count) if max_count else ""
        lines.append(f"  [{low:.2f}, {high:.2f}) {int(count):>12d}  {bar}")

    lines.append("")
    lines.append(f"Total pairs in histogram: {total}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote histogram: %s", path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always write a header, even if empty.
    if df.empty:
        columns = [
            "canonical_root_id",
            "canonical_root_label",
            "merged_id",
            "merged_label",
            "cosine_sim",
            "n_occurrences_root",
            "n_occurrences_merged",
            "shares_substring",
            "shares_token",
            "name_containment",
            "exact_normalized_match",
            "root_ner_noise_flag",
            "merged_ner_noise_flag",
            "suggested_action",
            "review_priority",
            "reasoning",
            "accepted",
        ]
        df = pd.DataFrame(columns=columns)

    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote proposals CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    log.info("Wrote stats JSON: %s", path)


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
        help=f"Path to graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Path to layer_2_embeddings.npz. Default: {DEFAULT_EMBEDDINGS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=DEFAULT_MIN_SIM,
        help="Minimum cosine similarity to consider a pair.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per similarity chunk. Lower this if memory is tight.",
    )
    parser.add_argument(
        "--histogram-buckets",
        type=int,
        default=20,
        help="Number of buckets for pairwise similarity histogram.",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        help="Optional maximum number of proposals to write.",
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

    if not (0.0 <= args.min_sim <= 1.0):
        log.error("--min-sim must be between 0 and 1")
        return 2
    if args.chunk_size < 1:
        log.error("--chunk-size must be >= 1")
        return 2
    if args.histogram_buckets < 2:
        log.error("--histogram-buckets must be >= 2")
        return 2
    if args.max_proposals is not None and args.max_proposals < 1:
        log.error("--max-proposals must be >= 1")
        return 2

    try:
        ids, vectors = load_embeddings(args.embeddings, allow_pickle=args.allow_pickle_npz)
        vectors = normalize_vectors(vectors)

        metadata = load_graph_metadata(args.graph)

        missing_meta = sorted(set(ids.astype(str)) - set(metadata))
        if missing_meta:
            log.warning(
                "%d ingredient embeddings have no matching Ingredient metadata in graph",
                len(missing_meta),
            )

        proposals_df, histogram, counts = compute_pair_proposals(
            vectors,
            ids,
            metadata,
            min_sim=args.min_sim,
            chunk_size=args.chunk_size,
            histogram_buckets=args.histogram_buckets,
            max_proposals=args.max_proposals,
        )

        out_csv = args.output_dir / "ingredient_canonicalization_proposals.csv"
        out_hist = args.output_dir / "ingredient_similarity_histogram.txt"
        out_stats = args.output_dir / "ingredient_canonicalization_stats.json"

        write_csv(proposals_df, out_csv)
        write_histogram(histogram, out_hist)

        stats = {
            "inputs": {
                "graph": str(args.graph),
                "embeddings": str(args.embeddings),
            },
            "outputs": {
                "proposals_csv": str(out_csv),
                "histogram_txt": str(out_hist),
                "stats_json": str(out_stats),
            },
            "thresholds": {
                "min_sim": args.min_sim,
                "auto_safe_sim": AUTO_SAFE_SIM,
                "auto_safe_with_substring": AUTO_SAFE_WITH_SUBSTRING,
                "review_floor_with_substring": REVIEW_FLOOR_WITH_SUBSTRING,
            },
            "parameters": {
                "chunk_size": args.chunk_size,
                "histogram_buckets": args.histogram_buckets,
                "max_proposals": args.max_proposals,
                "allow_pickle_npz": args.allow_pickle_npz,
            },
            "counts": counts,
            "notes": [
                "This step only proposes candidate merges; it does not modify the graph.",
                "The accepted column is intentionally empty and must be filled by a reviewer.",
                "Very high semantic similarity without lexical overlap remains review to protect cross-lingual fragmentation signals.",
            ],
        }

        write_json(out_stats, stats)

        log.info("Counts: %s", counts)
        log.info("Done. Next: review the CSV and fill accepted with TRUE/FALSE.")

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        pickle.PickleError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
