"""

======================

Build a contemporary recipe block from RecipeNLG-like data.

The script can load RecipeNLG from Hugging Face or read a local CSV/JSON/JSONL/
Parquet file with RecipeNLG-style columns. It classifies recipes into broad
European culinary regions using transparent ingredient-marker signatures, then
exports sampled records in the same JSON schema expected by build_graph_step1.py.

The goal is not to assign authoritative cuisines. The classifier is a
high-precision heuristic used to create a geographically diverse contemporary
comparison block. Recipes without enough marker evidence are left unclassified.

Default output
--------------

  data/contemporary_recipes.json

Typical usage
-------------

Load from Hugging Face:

  python ingest_contemporary.py --per-region 500 --output data/contemporary_recipes.json

Use a local file instead:

  python ingest_contemporary.py \
    --input-file data/recipenlg_sample.parquet \
    --per-region 300 \
    --output data/contemporary_recipes.json

Preview classification counts without writing records:

  python ingest_contemporary.py --dry-run

Merge with historical data:

  python -c "
  import json
  hist = json.load(open('data/relish_dataset.json', encoding='utf-8'))
  cont = json.load(open('data/contemporary_recipes.json', encoding='utf-8'))
  json.dump(hist + cont, open('data/relish_full.json', 'w', encoding='utf-8'),
            ensure_ascii=False, indent=1)
  "

Then rebuild:

  python build_graph_step1.py --input data/relish_full.json --output-dir data_full/

Dependencies
------------

  pip install pandas
  pip install datasets pyarrow     # only needed for Hugging Face / Parquet input

Safety / reproducibility
------------------------

This script does not use eval(). Ingredient and direction fields are parsed with
ast.literal_eval when they are stored as stringified Python lists.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT = Path("data/contemporary_recipes.json")
DEFAULT_HF_DATASET = "recipe_nlg"
DEFAULT_HF_SPLIT = "train"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("ingest_contemporary")


# ---------------------------------------------------------------------------
# European culinary-region signatures
# ---------------------------------------------------------------------------

REGION_SIGNATURES: dict[str, dict[str, Any]] = {
    "mediterranean_west": {
        "place": "Mediterranean West",
        "countries": "Spain, Southern France, Portugal",
        "markers": {
            "olive oil", "saffron", "paprika", "chorizo", "sherry",
            "manchego", "pimiento", "smoked paprika", "serrano",
            "bacalao", "sofrito", "romesco", "aioli", "alioli",
            "piquillo", "membrillo", "cava", "fino", "jerez",
            "piment d'espelette", "herbes de provence", "pastis",
        },
    },
    "italian": {
        "place": "Italy",
        "countries": "Italy",
        "markers": {
            "parmesan", "parmigiano", "pecorino", "ricotta",
            "mozzarella", "mascarpone", "prosciutto", "pancetta",
            "basil", "oregano", "balsamic", "balsamic vinegar",
            "risotto", "arborio", "polenta", "focaccia",
            "marinara", "pesto", "ciabatta", "gnocchi",
            "amaretti", "marsala", "grappa", "prosecco",
        },
    },
    "french": {
        "place": "France",
        "countries": "France",
        "markers": {
            "dijon", "dijon mustard", "gruyere", "emmental",
            "brie", "camembert", "creme fraiche", "crème fraîche",
            "tarragon", "chervil", "shallot", "shallots",
            "cognac", "calvados", "armagnac", "pernod",
            "beurre", "roux", "bechamel", "béchamel",
            "bouquet garni", "fines herbes", "fleur de sel",
        },
    },
    "british_irish": {
        "place": "British Isles",
        "countries": "United Kingdom, Ireland",
        "markers": {
            "worcestershire", "marmite", "stilton", "cheddar",
            "clotted cream", "double cream", "golden syrup",
            "treacle", "custard powder", "suet", "lard",
            "malt vinegar", "brown sauce", "piccalilli",
            "bramley", "swede", "turnip", "parsnip",
            "colman", "hp sauce",
        },
    },
    "germanic": {
        "place": "Central Europe",
        "countries": "Germany, Austria, Switzerland",
        "markers": {
            "sauerkraut", "bratwurst", "spatzle", "spätzle",
            "pumpernickel", "pretzel", "juniper", "caraway",
            "horseradish", "quark", "kirsch", "schnapps",
            "riesling", "knödel", "strudel", "wiener",
            "spaetzle", "lebkuchen", "marzipan",
        },
    },
    "nordic": {
        "place": "Scandinavia",
        "countries": "Denmark, Sweden, Norway, Finland",
        "markers": {
            "dill", "lingonberry", "juniper berry", "cardamom",
            "aquavit", "gravlax", "herring", "pickled herring",
            "rye bread", "crispbread", "cloudberry",
            "smoked salmon", "beetroot", "crayfish",
        },
    },
    "eastern_european": {
        "place": "Eastern Europe",
        "countries": "Poland, Hungary, Czech Republic, Romania",
        "markers": {
            "sour cream", "paprika", "caraway seed", "dill",
            "beetroot", "pierogi", "kielbasa", "sauerkraut",
            "poppy seed", "cottage cheese", "goulash",
            "plum", "slivovitz", "horseradish",
        },
    },
    "greek_turkish": {
        "place": "Eastern Mediterranean",
        "countries": "Greece, Turkey, Cyprus",
        "markers": {
            "feta", "phyllo", "filo", "tahini", "yogurt",
            "sumac", "pomegranate", "pomegranate molasses",
            "ouzo", "raki", "halloumi", "oregano",
            "olive", "kalamata", "za'atar", "labneh",
            "bulgur", "freekeh", "mint", "pita",
        },
    },
}


# ---------------------------------------------------------------------------
# Ingredient parsing
# ---------------------------------------------------------------------------

_QTY_RE = re.compile(
    r"^[\d\s/.½¼¾⅓⅔⅛⅜⅝⅞-]*"
    r"\s*"
    r"(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lbs?|"
    r"grams?|g|kg|ml|milliliters?|millilitres?|liters?|litres?|"
    r"quarts?|pints?|gallons?|cans?|packages?|pkg|sticks?|slices?|"
    r"pieces?|heads?|cloves?|bunch(?:es)?|sprigs?|pinch(?:es)?|dash(?:es)?|"
    r"handfuls?|small|medium|large|inch(?:es)?|cm)\b"
    r"\s*(?:of\s+)?",
    re.IGNORECASE,
)

_PREP_SUFFIXES = re.compile(
    r",?\s*(?:chopped|diced|minced|sliced|grated|shredded|crushed|"
    r"ground|melted|softened|divided|optional|to taste|"
    r"finely|coarsely|roughly|thinly|freshly|"
    r"at room temperature|peeled|seeded|cored|trimmed|"
    r"drained|rinsed|cooked|uncooked|frozen|thawed|"
    r"packed|sifted|toasted|dried|fresh)\b.*$",
    re.IGNORECASE,
)

_PARENS_RE = re.compile(r"\([^)]*\)")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_ingredient_name(raw: str) -> str:
    """Extract a core ingredient name from a RecipeNLG ingredient string.

    Examples:
      '1 1/2 cups all-purpose flour, sifted' -> 'all purpose flour'
      '2 cloves garlic, minced'              -> 'garlic'
    """
    s = normalize_text(str(raw))
    s = _PARENS_RE.sub("", s).strip()
    s = _QTY_RE.sub("", s).strip()
    s = _PREP_SUFFIXES.sub("", s).strip()
    s = s.strip(" ,.-;:")
    s = normalize_text(s)

    # Conservative singular normalization for common plural endings.
    # Avoid aggressive stemming; graph construction can handle variants later.
    replacements = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "leaves": "leaf",
        "loaves": "loaf",
    }
    return replacements.get(s, s)


def coerce_sequence(value: Any) -> list[str]:
    """Convert common RecipeNLG list encodings to a list of strings.

    The original RecipeNLG fields are often lists, but CSV exports may store
    them as stringified Python lists. This uses ast.literal_eval, not eval.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value if x is not None]

    if isinstance(value, tuple):
        return [str(x) for x in value if x is not None]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None]
            except (ValueError, SyntaxError):
                return []

        # Fallback for simple line-separated or pipe-separated strings.
        if "\n" in text:
            return [x.strip() for x in text.splitlines() if x.strip()]
        if "|" in text:
            return [x.strip() for x in text.split("|") if x.strip()]

        return [text]

    return []


def ingredients_to_relish(raw_list: list[str], confidence: float = 0.85) -> list[dict[str, Any]]:
    """Convert ingredient strings to the RELISH ingredient schema."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in raw_list:
        name = parse_ingredient_name(raw)

        if not name or len(name) < 2:
            continue
        if name in seen:
            continue

        seen.add(name)
        result.append({
            "name": name,
            "confidence_score": confidence,
            "specific_forms": [str(raw).strip()],
        })

    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def load_region_signatures(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional region signatures from JSON, otherwise use defaults."""
    if path is None:
        return REGION_SIGNATURES

    if not path.exists():
        raise FileNotFoundError(f"Region signatures JSON not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Region signatures file must contain a JSON object")

    for region, info in data.items():
        if not isinstance(info, dict) or "markers" not in info or "place" not in info:
            raise ValueError(f"Invalid region signature for {region!r}: expected place and markers")

        markers = info["markers"]
        if not isinstance(markers, (list, set, tuple)):
            raise ValueError(f"Invalid markers for {region!r}: expected list/set/tuple")

        info["markers"] = {normalize_text(str(x)) for x in markers if str(x).strip()}

    return data


def classify_recipe(
    ingredient_names: set[str],
    *,
    signatures: dict[str, dict[str, Any]],
    min_markers: int,
) -> tuple[str | None, int, dict[str, int]]:
    """Return (region_id, n_matches, all_scores)."""
    scores: dict[str, int] = {}

    for region_id, info in signatures.items():
        markers = {normalize_text(str(marker)) for marker in info["markers"]}
        count = 0

        for marker in markers:
            if marker in ingredient_names:
                count += 1
                continue

            # Substring matching is intentionally conservative enough for
            # multiword markers such as "olive oil" or "dijon mustard".
            for ingredient in ingredient_names:
                if marker in ingredient or ingredient in marker:
                    count += 1
                    break

        scores[region_id] = count

    if not scores:
        return None, 0, {}

    best_region, best_count = max(scores.items(), key=lambda item: (item[1], item[0]))

    if best_count >= min_markers:
        return best_region, best_count, scores

    return None, 0, scores


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_huggingface_dataset(
    dataset_name: str,
    *,
    split: str,
    data_dir: Path | None,
    cache_dir: Path | None,
    trust_remote_code: bool,
) -> pd.DataFrame:
    """Load a Hugging Face dataset into a DataFrame."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' library: pip install datasets") from exc

    kwargs: dict[str, Any] = {
        "split": split,
    }
    if data_dir is not None:
        kwargs["data_dir"] = str(data_dir)
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    log.info("Loading Hugging Face dataset %r split=%r", dataset_name, split)
    dataset = load_dataset(dataset_name, **kwargs)
    df = dataset.to_pandas()
    log.info("Loaded %d rows from Hugging Face", len(df))
    return df


def load_local_file(path: Path) -> pd.DataFrame:
    """Load CSV, JSON, JSONL, or Parquet into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    log.info("Loading local input file: %s", path)
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        df = pd.DataFrame(data)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input file type: {suffix}")

    log.info("Loaded %d rows from %s", len(df), path)
    return df


def validate_columns(df: pd.DataFrame, title_col: str, ingredients_col: str, directions_col: str) -> None:
    required = {title_col, ingredients_col, directions_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def build_recipe_text(title: str, ingredients: list[str], directions: list[str]) -> str:
    parts = [title.strip()]

    if ingredients:
        parts.append("Ingredients:\n" + "\n".join(f"- {item}" for item in ingredients))

    if directions:
        parts.append("Directions:\n" + "\n".join(f"{i + 1}. {step}" for i, step in enumerate(directions)))

    return "\n\n".join(part for part in parts if part.strip())


def process_dataset(
    df: pd.DataFrame,
    *,
    signatures: dict[str, dict[str, Any]],
    per_region: int,
    min_markers: int,
    seed: int,
    title_col: str,
    ingredients_col: str,
    directions_col: str,
    source_col: str | None,
    source_year: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify, sample, and convert recipes to RELISH-compatible records."""
    validate_columns(df, title_col, ingredients_col, directions_col)

    if per_region < 1:
        raise ValueError("--per-region must be >= 1")
    if min_markers < 1:
        raise ValueError("--min-markers must be >= 1")

    log.info("Parsing ingredients and classifying recipes")

    classified_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for df_idx, row in df.iterrows():
        raw_ingredients = coerce_sequence(row.get(ingredients_col))
        if not raw_ingredients:
            stats["empty_ingredient_field"] += 1
            continue

        parsed = [parse_ingredient_name(item) for item in raw_ingredients]
        ingredient_set = {item for item in parsed if item}

        if not ingredient_set:
            stats["no_parsed_ingredients"] += 1
            continue

        region, n_matches, scores = classify_recipe(
            ingredient_set,
            signatures=signatures,
            min_markers=min_markers,
        )

        if region is None:
            stats["unclassified"] += 1
            continue

        classified_rows.append({
            "df_idx": int(df_idx),
            "region": region,
            "n_matches": int(n_matches),
            "scores": scores,
        })

    classified_df = pd.DataFrame(classified_rows)

    if classified_df.empty:
        log.warning("No recipes met the region-classification threshold")
        return [], {
            "input_rows": int(len(df)),
            "records_generated": 0,
            "classification_counts": {},
            "stats": dict(stats),
        }

    classification_counts = {
        region: int(count)
        for region, count in classified_df["region"].value_counts().items()
    }

    log.info(
        "Classified %d recipes into European regions (%.2f%% of input)",
        len(classified_df),
        100 * len(classified_df) / max(1, len(df)),
    )

    for region, count in classification_counts.items():
        place = signatures[region].get("place", region)
        log.info("  %s (%s): %d recipes", region, place, count)

    sampled_indices: list[int] = []

    for region in signatures:
        region_rows = classified_df[classified_df["region"] == region].copy()

        if region_rows.empty:
            log.warning("No recipes found for region %s", region)
            continue

        # Deterministic sample among high-scoring rows. Sort by n_matches first,
        # then sample with fixed seed if more rows than requested have the same
        # score band.
        region_rows = region_rows.sort_values(
            by=["n_matches", "df_idx"],
            ascending=[False, True],
            kind="mergesort",
        )

        n = min(per_region, len(region_rows))

        # Keep strongest matches. This is deterministic and transparent.
        sample = region_rows.head(n)
        sampled_indices.extend(sample["df_idx"].astype(int).tolist())
        log.info("Sampled %d from %s", n, region)

    # Preserve deterministic order by region order, then original DataFrame index.
    sampled_set = set(sampled_indices)
    classification_by_idx = {
        int(row["df_idx"]): row
        for row in classified_rows
        if int(row["df_idx"]) in sampled_set
    }

    records: list[dict[str, Any]] = []
    output_stats: Counter[str] = Counter()

    for df_idx in sampled_indices:
        row = df.loc[df_idx]
        classification = classification_by_idx.get(int(df_idx))
        if not classification:
            continue

        region = classification["region"]
        region_info = signatures[region]

        raw_ingredients = coerce_sequence(row.get(ingredients_col))
        raw_directions = coerce_sequence(row.get(directions_col))

        ingredients = ingredients_to_relish(raw_ingredients, confidence=confidence)
        if not ingredients:
            output_stats["no_ingredients_after_parse"] += 1
            continue

        title = str(row.get(title_col) or "Untitled").strip() or "Untitled"
        recipe_text = build_recipe_text(title, raw_ingredients, raw_directions)

        source_author = "RecipeNLG"
        if source_col and source_col in df.columns:
            source_author = str(row.get(source_col) or "RecipeNLG")

        record = {
            "source_id": f"recipenlg_{region}",
            "source_title": "RecipeNLG",
            "source_author": source_author,
            "source_year": source_year,
            "source_place": region_info.get("place", region),
            "source_language": "en",
            "title": title,
            "recipe_text": recipe_text,
            "translation": recipe_text,
            "ingredients": ingredients,
            "tools": [],
            "actions": [],
            "classification": {
                "method": "ingredient_marker_signature",
                "region": region,
                "marker_matches": int(classification["n_matches"]),
                "region_place": region_info.get("place", region),
                "region_countries": region_info.get("countries"),
            },
        }

        records.append(record)
        output_stats[region] += 1

    summary = {
        "input_rows": int(len(df)),
        "classified_rows": int(len(classified_df)),
        "records_generated": int(len(records)),
        "classification_counts": classification_counts,
        "sampled_counts": dict(output_stats),
        "stats": dict(stats),
        "parameters": {
            "per_region": per_region,
            "min_markers": min_markers,
            "seed": seed,
            "source_year": source_year,
            "confidence": confidence,
        },
    }

    return records, summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    log.info("Wrote %s", path)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nSUMMARY")
    print(f"  input rows        : {summary['input_rows']}")
    print(f"  classified rows   : {summary.get('classified_rows', 0)}")
    print(f"  records generated : {summary['records_generated']}")
    print("\nSampled counts:")
    for region, count in sorted(summary.get("sampled_counts", {}).items()):
        print(f"  {region:25s} {count:6d}")

    if summary.get("stats"):
        print("\nOther stats:")
        for key, count in sorted(summary["stats"].items()):
            print(f"  {key:25s} {count:6d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--input-file",
        type=Path,
        help="Local CSV/JSON/JSONL/Parquet file. If omitted, Hugging Face is used.",
    )
    source_group.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help=f"Hugging Face dataset name. Default: {DEFAULT_HF_DATASET}",
    )

    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help=f"Hugging Face split. Default: {DEFAULT_HF_SPLIT}",
    )
    parser.add_argument(
        "--hf-data-dir",
        type=Path,
        help="Optional Hugging Face data_dir.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        help="Optional Hugging Face cache_dir.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face datasets.",
    )
    parser.add_argument(
        "--region-signatures",
        type=Path,
        help="Optional JSON file replacing built-in region signatures.",
    )
    parser.add_argument(
        "--per-region",
        type=int,
        default=500,
        help="Maximum recipes sampled per European region.",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=3,
        help="Minimum marker ingredients required for region classification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed reserved for future sampling variants; output is deterministic by default.",
    )
    parser.add_argument(
        "--source-year",
        type=int,
        default=2020,
        help="Approximate source_year assigned to contemporary recipes.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="Confidence score assigned to heuristically parsed ingredients.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for RELISH-compatible JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts but do not write the RELISH records.",
    )
    parser.add_argument(
        "--title-col",
        default="title",
        help="Column containing recipe title.",
    )
    parser.add_argument(
        "--ingredients-col",
        default="ingredients",
        help="Column containing ingredient list.",
    )
    parser.add_argument(
        "--directions-col",
        default="directions",
        help="Column containing directions/instructions list.",
    )
    parser.add_argument(
        "--source-col",
        default="source",
        help="Optional column used as source_author when present.",
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

    if args.per_region < 1:
        log.error("--per-region must be >= 1")
        return 2
    if args.min_markers < 1:
        log.error("--min-markers must be >= 1")
        return 2
    if args.confidence < 0 or args.confidence > 1:
        log.error("--confidence must be between 0 and 1")
        return 2

    try:
        signatures = load_region_signatures(args.region_signatures)

        if args.input_file:
            df = load_local_file(args.input_file)
        else:
            df = load_huggingface_dataset(
                args.hf_dataset,
                split=args.hf_split,
                data_dir=args.hf_data_dir,
                cache_dir=args.hf_cache_dir,
                trust_remote_code=args.trust_remote_code,
            )

        records, summary = process_dataset(
            df,
            signatures=signatures,
            per_region=args.per_region,
            min_markers=args.min_markers,
            seed=args.seed,
            title_col=args.title_col,
            ingredients_col=args.ingredients_col,
            directions_col=args.directions_col,
            source_col=args.source_col,
            source_year=args.source_year,
            confidence=args.confidence,
        )

        print_summary(summary)

        if args.out_summary:
            write_json(args.out_summary, summary)

        if args.dry_run:
            log.info("Dry run: not writing RELISH output records")
            return 0

        write_json(args.output, records)

        log.info("")
        log.info("NEXT STEPS:")
        log.info("  1. Merge with historical data.")
        log.info("  2. Rebuild the graph with build_graph_step1.py.")
        log.info("  3. Re-run embeddings, clustering, and analysis on the full graph.")

        return 0

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
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
