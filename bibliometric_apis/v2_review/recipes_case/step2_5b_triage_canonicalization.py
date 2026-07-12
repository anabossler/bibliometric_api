"""
===================================

Triage ingredient-canonicalization proposals from Step 2.5.

Step 2.5 can produce thousands of embedding-similarity proposals. This script
reduces those raw pairs into actionable bins using deterministic rules:

1. AUTO_MERGE_ANAGRAM
   Names contain the same multiset of words in different order:
   broth_wine <-> wine_broth.

2. NEEDS_DELETION
   One or both nodes look like NER junk: phrase fragments, function words,
   nutritional-table tokens, numeric measurement leakage, or nodes already
   flagged upstream with ner_noise_flag=True.

3. AUTO_MERGE_MORPHO
   Names differ only by simple morphology: plural, gerund/past participle,
   or short suffix/case variation on the same root.

4. NEEDS_REVIEW
   Remaining pairs above the review similarity threshold. These are grouped by
   hub so the expert reviews one root with many candidate variants at once.

5. DISCARDED
   Pairs below the review threshold and not captured by safer automatic rules.

This script does NOT modify the graph. It only creates audit/review files.

Reads
-----

  data/ingredient_canonicalization_proposals.csv
  data/graph_step1.gpickle

Writes
------

  data/canonicalization_auto_merge.csv
  data/canonicalization_needs_review.csv
  data/canonicalization_needs_review_pairs.csv
  data/canonicalization_deletion_candidates.csv
  data/canonicalization_discarded_pairs.csv        optional with --write-discarded
  data/canonicalization_triage_stats.json

Usage
-----

  python step2_5b_triage_canonicalization.py

  python step2_5b_triage_canonicalization.py \
    --proposals data/ingredient_canonicalization_proposals.csv \
    --graph data/graph_step1.gpickle \
    --output-dir data/

  python step2_5b_triage_canonicalization.py \
    --review-sim-threshold 0.92 \
    --min-phrase-words 4 \
    --auto-merge-morpho-min-sim 0.93

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROPOSALS = Path("data/ingredient_canonicalization_proposals.csv")
DEFAULT_GRAPH = Path("data/graph_step1.gpickle")
DEFAULT_OUTPUT_DIR = Path("data")

DEFAULT_REVIEW_SIM_THRESHOLD = 0.92
DEFAULT_MIN_PHRASE_WORDS = 4
DEFAULT_AUTO_MERGE_MORPHO_MIN_SIM = 0.93

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step2_5b_triage_canonicalization")


# ---------------------------------------------------------------------------
# Junk-token configuration
# ---------------------------------------------------------------------------

JUNK_STRONG = {
    # English determiners / pronouns
    "the", "that", "this", "those", "these", "his", "her", "its", "their",
    "any", "some", "each", "every",
    # Demonstratives / discourse
    "aforementioned", "above", "below", "following",
    # NER-leakage placeholders
    "drinking", "measure",
    # Nutritional table leakage
    "information", "composition", "nutrition", "nutritional", "calorie",
    "calories", "protein", "carbohydrate", "carbohydrates", "fat", "fiber",
    # German function words
    "der", "die", "das", "und", "von", "mit", "im", "zum",
    # Spanish / Catalan / Portuguese / French function words
    "el", "la", "los", "las", "de", "del", "da", "do", "dos", "das",
    "un", "una", "uns", "unes", "le", "les", "des", "du",
}

JUNK_STANDALONE = {
    # Generic nouns
    "kind", "type", "amount", "quantity", "portion", "preparation", "cookery",
    "recipe", "ingredient", "ingredients",
    # Bare verbs leaked from NER
    "cook", "cooking", "cooked",
    "bake", "baking", "baked",
    "blend", "blending", "blended",
    "boil", "boiling", "boiled",
    "mix", "mixing", "mixed",
    "stir", "stirring", "stirred",
    "serve", "serving", "served",
}


# ---------------------------------------------------------------------------
# Data structures and loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngredientMeta:
    node_id: str
    label: str
    n_occurrences: int
    ner_noise_flag: bool


def load_proposals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Proposals CSV not found: {path}")

    log.info("Loading proposals: %s", path)
    df = pd.read_csv(path)

    required = {"canonical_root_id", "merged_id", "cosine_sim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Proposals CSV is missing required columns: {sorted(missing)}")

    df["canonical_root_id"] = df["canonical_root_id"].astype(str)
    df["merged_id"] = df["merged_id"].astype(str)
    df["cosine_sim"] = pd.to_numeric(df["cosine_sim"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["cosine_sim"]).copy()
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d proposal rows with invalid cosine_sim", dropped)

    log.info("Loaded %d usable proposals", len(df))
    return df


def load_graph_metadata(path: Path) -> dict[str, IngredientMeta]:
    """Load Ingredient metadata from a trusted graph."""
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
            or name_of(node_id_str)
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

    log.info("Loaded metadata for %d Ingredient nodes", len(metadata))
    return metadata


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def name_of(node_id: str) -> str:
    return str(node_id).split("::", 1)[1] if "::" in str(node_id) else str(node_id)


def normalize_name(text: str) -> str:
    text = name_of(str(text))
    text = strip_accents(text).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def words_of(text: str) -> list[str]:
    return [word for word in normalize_name(text).split() if word]


def label_for(node_id: str, metadata: dict[str, IngredientMeta], fallback_label: str | None = None) -> str:
    meta = metadata.get(node_id)
    if meta:
        return meta.label
    if fallback_label and str(fallback_label).strip():
        return str(fallback_label)
    return name_of(node_id)


# ---------------------------------------------------------------------------
# Triage rules
# ---------------------------------------------------------------------------

def is_junk_node(
    node_id: str,
    *,
    metadata: dict[str, IngredientMeta],
    min_phrase_words: int,
    strong_junk: set[str],
    standalone_junk: set[str],
) -> tuple[bool, str]:
    """Return (is_junk, reason)."""
    meta = metadata.get(node_id)
    if meta and meta.ner_noise_flag:
        return True, "ner_noise_flag"

    words = words_of(meta.label if meta else node_id)

    if len(words) >= min_phrase_words:
        return True, f"phrase>={min_phrase_words}_words"

    if any(re.fullmatch(r"\d+\w*", word) for word in words):
        return True, "numeric_token"

    for word in words:
        if word in strong_junk:
            return True, f"junk_strong:{word}"

    if len(words) == 1 and words[0] in standalone_junk:
        return True, f"junk_standalone:{words[0]}"

    return False, ""


def is_anagram_pair(id_a: str, id_b: str, metadata: dict[str, IngredientMeta]) -> bool:
    """True if normalized names have the same multiset of words."""
    words_a = words_of(metadata.get(id_a).label if id_a in metadata else id_a)
    words_b = words_of(metadata.get(id_b).label if id_b in metadata else id_b)

    if not words_a or not words_b:
        return False
    if words_a == words_b:
        return False  # exact same order/name is handled elsewhere by Step 2.5
    return sorted(words_a) == sorted(words_b)


def simple_stem(word: str) -> str:
    """Small deterministic stemmer for obvious culinary NER variants."""
    word = word.lower()

    replacements = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "leaves": "leaf",
        "loaves": "loaf",
    }
    if word in replacements:
        return replacements[word]

    suffixes = [
        "ingly", "edly", "ing", "ed", "es", "s",
        # German / Romance case or plural-ish endings, conservative
        "en", "er", "em", "e",
    ]

    for suffix in suffixes:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]

    return word


def morpho_match_word(word_a: str, word_b: str) -> bool:
    if word_a == word_b:
        return True

    stem_a = simple_stem(word_a)
    stem_b = simple_stem(word_b)

    if stem_a == stem_b and len(stem_a) >= 3:
        return True

    if abs(len(word_a) - len(word_b)) > 3:
        return False

    common = 0
    for char_a, char_b in zip(word_a, word_b):
        if char_a == char_b:
            common += 1
        else:
            break

    min_len = min(len(word_a), len(word_b))
    return min_len >= 3 and common >= 3 and common >= min_len * 0.70


def is_morpho_pair(id_a: str, id_b: str, metadata: dict[str, IngredientMeta]) -> bool:
    words_a = words_of(metadata.get(id_a).label if id_a in metadata else id_a)
    words_b = words_of(metadata.get(id_b).label if id_b in metadata else id_b)

    if not words_a or not words_b:
        return False

    if len(words_a) != len(words_b):
        return False

    return all(morpho_match_word(a, b) for a, b in zip(words_a, words_b))


def classify_row(
    row: pd.Series,
    *,
    metadata: dict[str, IngredientMeta],
    review_sim_threshold: float,
    min_phrase_words: int,
    auto_merge_morpho_min_sim: float,
    strong_junk: set[str],
    standalone_junk: set[str],
    anagram_before_deletion: bool,
) -> tuple[str, str]:
    """Return (category, reason) for one proposal row."""
    root_id = str(row["canonical_root_id"])
    merged_id = str(row["merged_id"])
    sim = float(row["cosine_sim"])

    if anagram_before_deletion and is_anagram_pair(root_id, merged_id, metadata):
        return "auto_merge_anagram", "same_word_multiset"

    root_junk, root_reason = is_junk_node(
        root_id,
        metadata=metadata,
        min_phrase_words=min_phrase_words,
        strong_junk=strong_junk,
        standalone_junk=standalone_junk,
    )
    merged_junk, merged_reason = is_junk_node(
        merged_id,
        metadata=metadata,
        min_phrase_words=min_phrase_words,
        strong_junk=strong_junk,
        standalone_junk=standalone_junk,
    )

    if root_junk or merged_junk:
        reasons: list[str] = []
        if root_junk:
            reasons.append(f"root:{root_reason}")
        if merged_junk:
            reasons.append(f"merged:{merged_reason}")
        return "needs_deletion", "|".join(reasons)

    if not anagram_before_deletion and is_anagram_pair(root_id, merged_id, metadata):
        return "auto_merge_anagram", "same_word_multiset"

    if sim >= auto_merge_morpho_min_sim and is_morpho_pair(root_id, merged_id, metadata):
        return "auto_merge_morpho", "morphological_variant"

    if sim >= review_sim_threshold:
        return "needs_review", "above_review_threshold_no_auto_rule"

    return "discarded", f"sim<{review_sim_threshold:.3f}_and_no_auto_rule"


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def enrich_with_metadata(df: pd.DataFrame, metadata: dict[str, IngredientMeta]) -> pd.DataFrame:
    """Add labels/counts/flags if missing or stale."""
    out = df.copy()

    def meta_value(node_id: str, attr: str, default: Any = "") -> Any:
        meta = metadata.get(str(node_id))
        return getattr(meta, attr) if meta else default

    out["canonical_root_label"] = [
        label_for(str(node_id), metadata, out.iloc[i].get("canonical_root_label"))
        for i, node_id in enumerate(out["canonical_root_id"])
    ]
    out["merged_label"] = [
        label_for(str(node_id), metadata, out.iloc[i].get("merged_label"))
        for i, node_id in enumerate(out["merged_id"])
    ]

    out["n_occurrences_root"] = [
        int(meta_value(node_id, "n_occurrences", out.iloc[i].get("n_occurrences_root", 0) or 0))
        for i, node_id in enumerate(out["canonical_root_id"])
    ]
    out["n_occurrences_merged"] = [
        int(meta_value(node_id, "n_occurrences", out.iloc[i].get("n_occurrences_merged", 0) or 0))
        for i, node_id in enumerate(out["merged_id"])
    ]

    out["root_ner_noise_flag"] = [
        bool(meta_value(node_id, "ner_noise_flag", out.iloc[i].get("root_ner_noise_flag", False)))
        for i, node_id in enumerate(out["canonical_root_id"])
    ]
    out["merged_ner_noise_flag"] = [
        bool(meta_value(node_id, "ner_noise_flag", out.iloc[i].get("merged_ner_noise_flag", False)))
        for i, node_id in enumerate(out["merged_id"])
    ]

    return out


def collect_deletion_nodes(
    deletion_df: pd.DataFrame,
    *,
    metadata: dict[str, IngredientMeta],
    min_phrase_words: int,
    strong_junk: set[str],
    standalone_junk: set[str],
) -> pd.DataFrame:
    """Collect unique junk nodes from deletion pairs."""
    rows_by_node: dict[str, dict[str, Any]] = {}

    for _, row in deletion_df.iterrows():
        for role, column in [("root", "canonical_root_id"), ("merged", "merged_id")]:
            node_id = str(row[column])
            is_junk, reason = is_junk_node(
                node_id,
                metadata=metadata,
                min_phrase_words=min_phrase_words,
                strong_junk=strong_junk,
                standalone_junk=standalone_junk,
            )
            if not is_junk:
                continue

            meta = metadata.get(node_id)
            existing = rows_by_node.get(node_id)

            if existing:
                existing["n_pairs_involved"] += 1
                existing["roles"] = ",".join(sorted(set(existing["roles"].split(",")) | {role}))
                continue

            rows_by_node[node_id] = {
                "node_id": node_id,
                "label": meta.label if meta else name_of(node_id),
                "reason": reason,
                "n_occurrences": meta.n_occurrences if meta else 0,
                "ner_noise_flag": meta.ner_noise_flag if meta else False,
                "n_pairs_involved": 1,
                "roles": role,
                "accepted_for_deletion": "",
            }

    rows = list(rows_by_node.values())
    if not rows:
        return pd.DataFrame(columns=[
            "node_id",
            "label",
            "reason",
            "n_occurrences",
            "ner_noise_flag",
            "n_pairs_involved",
            "roles",
            "accepted_for_deletion",
        ])

    return pd.DataFrame(rows).sort_values(
        by=["reason", "n_occurrences", "node_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def group_review_by_hub(review_df: pd.DataFrame) -> pd.DataFrame:
    """Group review pairs so the expert reviews one hub at a time."""
    if review_df.empty:
        return pd.DataFrame(columns=[
            "hub_id",
            "hub_label",
            "n_candidates",
            "candidates_summary",
            "max_sim",
            "min_sim",
            "accepted_ids",
            "notes",
        ])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for _, row in review_df.iterrows():
        grouped[str(row["canonical_root_id"])].append({
            "candidate_id": str(row["merged_id"]),
            "candidate_label": str(row.get("merged_label", row["merged_id"])),
            "sim": float(row["cosine_sim"]),
            "reason": str(row.get("category_reason", "")),
        })

    rows: list[dict[str, Any]] = []
    for hub_id, candidates in grouped.items():
        candidates.sort(key=lambda item: (-item["sim"], item["candidate_id"]))
        sims = [item["sim"] for item in candidates]
        hub_label = str(review_df.loc[review_df["canonical_root_id"] == hub_id, "canonical_root_label"].iloc[0])

        candidate_summary = "; ".join(
            f"{item['candidate_id']} [{item['candidate_label']}] ({item['sim']:.4f})"
            for item in candidates
        )

        rows.append({
            "hub_id": hub_id,
            "hub_label": hub_label,
            "n_candidates": len(candidates),
            "candidates_summary": candidate_summary,
            "max_sim": max(sims),
            "min_sim": min(sims),
            "accepted_ids": "",
            "notes": "",
        })

    return pd.DataFrame(rows).sort_values(
        by=["n_candidates", "max_sim", "hub_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )


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


def load_extra_tokens(items: list[str], path: Path | None) -> set[str]:
    tokens = {normalize_name(item) for item in items or [] if normalize_name(item)}

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")
        tokens.update(
            normalize_name(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    return {token for token in tokens if token}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--proposals",
        type=Path,
        default=DEFAULT_PROPOSALS,
        help=f"Step 2.5 proposals CSV. Default: {DEFAULT_PROPOSALS}",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Step 1 graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--review-sim-threshold",
        type=float,
        default=DEFAULT_REVIEW_SIM_THRESHOLD,
        help="Pairs below this threshold are discarded unless caught by auto rules.",
    )
    parser.add_argument(
        "--min-phrase-words",
        type=int,
        default=DEFAULT_MIN_PHRASE_WORDS,
        help="Names with at least this many words are treated as phrase leakage.",
    )
    parser.add_argument(
        "--auto-merge-morpho-min-sim",
        type=float,
        default=DEFAULT_AUTO_MERGE_MORPHO_MIN_SIM,
        help="Minimum similarity for the morphological auto-merge rule.",
    )
    parser.add_argument(
        "--strong-junk-token",
        action="append",
        default=[],
        help="Additional strong junk token. Can be repeated.",
    )
    parser.add_argument(
        "--standalone-junk-token",
        action="append",
        default=[],
        help="Additional standalone-only junk token. Can be repeated.",
    )
    parser.add_argument(
        "--strong-junk-file",
        type=Path,
        help="Optional file with one additional strong junk token per line.",
    )
    parser.add_argument(
        "--standalone-junk-file",
        type=Path,
        help="Optional file with one additional standalone junk token per line.",
    )
    parser.add_argument(
        "--deletion-before-anagram",
        action="store_true",
        help=(
            "Apply deletion before anagram. Default keeps anagram first to avoid "
            "deleting token-noisy but symmetric variants such as baked_good/bake_good."
        ),
    )
    parser.add_argument(
        "--write-discarded",
        action="store_true",
        help="Also write canonicalization_discarded_pairs.csv.",
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

    if not (0.0 <= args.review_sim_threshold <= 1.0):
        log.error("--review-sim-threshold must be between 0 and 1")
        return 2
    if args.min_phrase_words < 2:
        log.error("--min-phrase-words must be >= 2")
        return 2
    if not (0.0 <= args.auto_merge_morpho_min_sim <= 1.0):
        log.error("--auto-merge-morpho-min-sim must be between 0 and 1")
        return 2

    try:
        proposals = load_proposals(args.proposals)
        metadata = load_graph_metadata(args.graph)
        proposals = enrich_with_metadata(proposals, metadata)

        strong_junk = set(JUNK_STRONG) | load_extra_tokens(args.strong_junk_token, args.strong_junk_file)
        standalone_junk = set(JUNK_STANDALONE) | load_extra_tokens(
            args.standalone_junk_token,
            args.standalone_junk_file,
        )

        log.info("Classifying proposals")
        categories: list[str] = []
        reasons: list[str] = []

        for _, row in proposals.iterrows():
            category, reason = classify_row(
                row,
                metadata=metadata,
                review_sim_threshold=args.review_sim_threshold,
                min_phrase_words=args.min_phrase_words,
                auto_merge_morpho_min_sim=args.auto_merge_morpho_min_sim,
                strong_junk=strong_junk,
                standalone_junk=standalone_junk,
                anagram_before_deletion=not args.deletion_before_anagram,
            )
            categories.append(category)
            reasons.append(reason)

        proposals["category"] = categories
        proposals["category_reason"] = reasons

        counts = {
            str(k): int(v)
            for k, v in proposals["category"].value_counts().sort_index().items()
        }
        log.info("Triage counts: %s", counts)

        auto_merge = proposals[
            proposals["category"].isin(["auto_merge_anagram", "auto_merge_morpho"])
        ].copy()

        needs_review_pairs = proposals[proposals["category"] == "needs_review"].copy()
        deletion_pairs = proposals[proposals["category"] == "needs_deletion"].copy()
        discarded_pairs = proposals[proposals["category"] == "discarded"].copy()

        deletion_nodes = collect_deletion_nodes(
            deletion_pairs,
            metadata=metadata,
            min_phrase_words=args.min_phrase_words,
            strong_junk=strong_junk,
            standalone_junk=standalone_junk,
        )

        review_grouped = group_review_by_hub(needs_review_pairs)

        out_auto = args.output_dir / "canonicalization_auto_merge.csv"
        out_deletion = args.output_dir / "canonicalization_deletion_candidates.csv"
        out_review = args.output_dir / "canonicalization_needs_review.csv"
        out_review_pairs = args.output_dir / "canonicalization_needs_review_pairs.csv"
        out_discarded = args.output_dir / "canonicalization_discarded_pairs.csv"
        out_stats = args.output_dir / "canonicalization_triage_stats.json"

        write_csv(auto_merge, out_auto)
        write_csv(deletion_nodes, out_deletion)
        write_csv(review_grouped, out_review)
        write_csv(needs_review_pairs, out_review_pairs)

        if args.write_discarded:
            write_csv(discarded_pairs, out_discarded)

        stats = {
            "inputs": {
                "proposals": str(args.proposals),
                "graph": str(args.graph),
            },
            "config": {
                "review_sim_threshold": args.review_sim_threshold,
                "min_phrase_words": args.min_phrase_words,
                "auto_merge_morpho_min_sim": args.auto_merge_morpho_min_sim,
                "anagram_before_deletion": not args.deletion_before_anagram,
                "n_strong_junk_tokens": len(strong_junk),
                "n_standalone_junk_tokens": len(standalone_junk),
            },
            "triage_counts": counts,
            "outputs": {
                "auto_merge_pairs": len(auto_merge),
                "deletion_nodes_unique": len(deletion_nodes),
                "deletion_pairs": len(deletion_pairs),
                "review_hubs": len(review_grouped),
                "review_pairs": len(needs_review_pairs),
                "discarded_pairs": len(discarded_pairs),
                "auto_merge_csv": str(out_auto),
                "deletion_candidates_csv": str(out_deletion),
                "needs_review_csv": str(out_review),
                "needs_review_pairs_csv": str(out_review_pairs),
                "discarded_pairs_csv": str(out_discarded) if args.write_discarded else None,
            },
            "notes": [
                "This step only triages candidate canonicalization decisions; it does not modify the graph.",
                "Auto-merge files are audit-only unless a later merge step explicitly applies them.",
                "Deletion candidates require confirmation before graph deletion.",
                "Needs-review grouped file expects accepted_ids as comma-separated ids to merge into hub_id.",
            ],
        }
        write_json(out_stats, stats)

        log.info("Done.")
        log.info("Next: audit auto_merge, confirm deletion candidates, and fill accepted_ids in needs_review.")

        return 0

    except (
        FileNotFoundError,
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
