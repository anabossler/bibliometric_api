"""

==============================

Re-extract ingredients from contaminated RELISH corpora with a restrictive LLM
prompt.

The purpose of this step is to repair ingredient contamination in selected
corpora by extracting only ingredients literally supported by each recipe text.
The script writes full per-recipe outputs plus an audit trail and a resumable
checkpoint.

Default input
-------------

  data/relish_dataset_with_sefardies.json

Default outputs
---------------

  data/reextracted_ingredients.jsonl
  data/reextract_checkpoint.json
  data/reextract_audit.csv
  data/reextract_summary.json

Strategy
--------

  - Select recipes whose source_id is in the contaminated-source list.
  - Send recipe text to an LLM through OpenRouter.
  - Require JSON output with verbatim ingredient evidence.
  - Audit whether each verbatim form appears in the input text.
  - Resume from checkpoint if interrupted.

Security / reproducibility
--------------------------

  - No hardcoded local .env path.
  - Use OPENROUTER_API_KEY from the environment or --dotenv .env.
  - --dry-run does not require an API key and does not call APIs.
  - All paths, model, retry policy, and contaminated sources are CLI-configurable.

Usage
-----

  python step6_reextract_ingredients.py --dry-run --limit 20

  python step6_reextract_ingredients.py --dotenv .env --limit 100

  python step6_reextract_ingredients.py --source sefardies_es --rerun

  python step6_reextract_ingredients.py \
    --input-json data/relish_dataset_with_sefardies.json \
    --model google/gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT_JSON = Path("data/relish_dataset_with_sefardies.json")
DEFAULT_OUT_JSONL = Path("data/reextracted_ingredients.jsonl")
DEFAULT_CHECKPOINT = Path("data/reextract_checkpoint.json")
DEFAULT_AUDIT_CSV = Path("data/reextract_audit.csv")
DEFAULT_SUMMARY_JSON = Path("data/reextract_summary.json")

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_BATCH_CHECKPOINT_EVERY = 25
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 1500
DEFAULT_MAX_TEXT_CHARS = 12000

DEFAULT_HTTP_REFERER = None
DEFAULT_X_TITLE = "Relish-Ingredient-ReExtraction"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step6_reextract_ingredients")


# Corpora to re-extract, based on prior contamination audit.
DEFAULT_CONTAMINATED_SOURCES = {
    "apicius",
    "sefardies_es",
    "a_miscellany",
    "enseignements",
    "veneziano",
    "cuina_catalana",
    "fundacio_alicia_0",
    "fundacio_alicia_1",
    "fundacio_alicia_2",
    "fundacio_alicia_3",
    "fundacio_alicia_4",
    "fundacio_alicia_5",
    "fundacio_alicia_6",
    "fundacio_alicia_7",
    "fundacio_alicia_8",
    "fundacio_alicia_9",
    "fundacio_alicia_10",
    "fundacio_alicia_11",
    "fundacio_alicia_12",
    "fundacio_alicia_13",
    "fundacio_alicia_14",
    "fundacio_alicia_15",
    "fundacio_alicia_16",
    "harpestreng",
    "proper_newe",
    "viandier_1485",
}


SYSTEM_PROMPT = """You are an expert in historical culinary text analysis.

Extract ONLY ingredients that are literally mentioned in the recipe text.

Strict rules:
1. Only extract nouns that appear verbatim in the text, or in a direct translation already present in the text.
2. Do not infer, modernize, complete, or add ingredients that would merely make culinary sense.
3. Do not include cooking actions, tools, vessels, measurements, nutritional terms, or metadata.
4. Do not include nutritional metadata such as cholesterol, vitamins, calories, carbohydrate, sodium, protein, or fat.
5. If an ingredient is uncertain, omit it.
6. Keep canonical names lowercase and concise.
7. Each extracted ingredient must include a verbatim phrase copied from the input text.

Return only valid JSON with this exact structure:
{
  "ingredients": [
    {
      "name": "<canonical_name_lowercase>",
      "verbatim_form": "<exact_phrase_from_text>"
    }
  ]
}

If the text contains no ingredients, return:
{"ingredients": []}

Do not add commentary.
"""


# ---------------------------------------------------------------------------
# Environment and generic helpers
# ---------------------------------------------------------------------------

def load_dotenv_file(path: Path | None) -> None:
    """Load KEY=VALUE pairs from a .env file without requiring python-dotenv."""
    if path is None:
        return

    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")

    with path.open(encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

    log.info("Loaded environment variables from %s", path)


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Use environment or --dotenv .env.")
    return api_key


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_json_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Data loading and source selection
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    log.info("Loading dataset: %s", path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of recipe records in {path}")

    records = [row for row in data if isinstance(row, dict)]
    if len(records) != len(data):
        log.warning("Dropped %d non-dict records from dataset", len(data) - len(records))

    log.info("Loaded %d recipe records", len(records))
    return records


def load_sources(default_sources: set[str], source_args: list[str], source_file: Path | None) -> set[str]:
    sources = set(default_sources)

    for item in source_args or []:
        if item.strip():
            sources.add(item.strip())

    if source_file is not None:
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        for line in source_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.add(line)

    return sources


def recipe_text(recipe: dict[str, Any], *, text_fields: list[str], max_chars: int) -> str:
    parts: list[str] = []

    for field in text_fields:
        value = safe_str(recipe.get(field))
        if value:
            parts.append(value)

    text = normalize_space(" ".join(parts))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    return text


def select_targets(
    data: list[dict[str, Any]],
    *,
    contaminated_sources: set[str],
    text_fields: list[str],
    max_text_chars: int,
    min_text_chars: int,
) -> tuple[list[tuple[int, dict[str, Any], str]], Counter[str]]:
    targets: list[tuple[int, dict[str, Any], str]] = []
    stats: Counter[str] = Counter()

    for index, recipe in enumerate(data):
        source_id = safe_str(recipe.get("source_id"))
        if source_id not in contaminated_sources:
            stats["skipped_clean_source"] += 1
            continue

        text = recipe_text(recipe, text_fields=text_fields, max_chars=max_text_chars)
        if len(text) < min_text_chars:
            stats["skipped_short_text"] += 1
            continue

        targets.append((index, recipe, text))
        stats["selected"] += 1

    return targets, stats


# ---------------------------------------------------------------------------
# Checkpoint and audit
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> set[int]:
    if not path.exists():
        return set()

    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        log.warning("Checkpoint is malformed and will be ignored: %s", path)
        return set()

    done = data.get("done_indices", [])
    if not isinstance(done, list):
        log.warning("Checkpoint done_indices is not a list; ignoring")
        return set()

    return {int(item) for item in done}


def save_checkpoint(path: Path, done_indices: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "done_indices": sorted(done_indices),
        "n_done": len(done_indices),
        "updated_at_utc": timestamp_utc(),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_audit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "source_id",
        "title",
        "n_extracted",
        "n_verbatim_real",
        "n_verbatim_missing",
        "n_empty_verbatim",
        "model",
        "timestamp_utc",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote audit CSV: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# LLM call and response parsing
# ---------------------------------------------------------------------------

def call_openrouter(
    *,
    text: str,
    api_key: str,
    model: str,
    openrouter_url: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    http_referer: str | None,
    x_title: str | None,
) -> dict[str, Any] | None:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Recipe text:\n{text}"},
        ],
    }

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(openrouter_url, data=data, headers=headers)
            with urlopen(request, timeout=request_timeout_s) as response:
                raw = response.read().decode("utf-8")

            response_payload = json.loads(raw)
            content = safe_str(response_payload["choices"][0]["message"]["content"])
            return parse_llm_json(content)

        except HTTPError as exc:
            last_error = exc
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OpenRouter HTTP error attempt %d/%d: %s %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc.code,
                error_body,
                wait,
            )
            time.sleep(wait)

        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OpenRouter error attempt %d/%d: %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    log.error("OpenRouter call failed after %d retries: %s", max_retries, last_error)
    return None


def parse_llm_json(content: str) -> dict[str, Any]:
    cleaned = strip_json_fences(content)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start:end + 1])
        else:
            raise ValueError("LLM output is not valid JSON")

    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object")

    ingredients = parsed.get("ingredients", [])
    if not isinstance(ingredients, list):
        ingredients = []

    cleaned_ingredients: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in ingredients:
        if not isinstance(item, dict):
            continue

        name = safe_str(item.get("name")).lower()
        verbatim = safe_str(item.get("verbatim_form"))

        if not name or not verbatim:
            continue

        key = (name, verbatim.lower())
        if key in seen:
            continue
        seen.add(key)

        cleaned_ingredients.append({
            "name": name,
            "verbatim_form": verbatim,
        })

    return {"ingredients": cleaned_ingredients}


# ---------------------------------------------------------------------------
# Extraction audit
# ---------------------------------------------------------------------------

def audit_extraction(text: str, extracted: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return n_verbatim_real, n_verbatim_missing, n_empty_verbatim."""
    text_lower = text.lower()
    real = 0
    missing = 0
    empty = 0

    for ingredient in extracted:
        verbatim = safe_str(ingredient.get("verbatim_form")).lower()
        if not verbatim:
            empty += 1
            continue
        if verbatim in text_lower:
            real += 1
        else:
            missing += 1

    return real, missing, empty


def build_output_record(
    *,
    index: int,
    recipe: dict[str, Any],
    extracted: list[dict[str, Any]],
    text: str,
    model: str,
) -> dict[str, Any]:
    real, missing, empty = audit_extraction(text, extracted)

    return {
        "index": int(index),
        "source_id": recipe.get("source_id"),
        "title": recipe.get("title", ""),
        "model": model,
        "timestamp_utc": timestamp_utc(),
        "ingredients": extracted,
        "audit": {
            "n_extracted": int(len(extracted)),
            "n_verbatim_real": int(real),
            "n_verbatim_missing": int(missing),
            "n_empty_verbatim": int(empty),
        },
    }


def output_to_audit_row(record: dict[str, Any]) -> dict[str, Any]:
    audit = record.get("audit", {})
    return {
        "index": str(record.get("index", "")),
        "source_id": safe_str(record.get("source_id")),
        "title": safe_str(record.get("title"))[:80],
        "n_extracted": str(audit.get("n_extracted", 0)),
        "n_verbatim_real": str(audit.get("n_verbatim_real", 0)),
        "n_verbatim_missing": str(audit.get("n_verbatim_missing", 0)),
        "n_empty_verbatim": str(audit.get("n_empty_verbatim", 0)),
        "model": safe_str(record.get("model")),
        "timestamp_utc": safe_str(record.get("timestamp_utc")),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def run_reextraction(args: argparse.Namespace) -> int:
    load_dotenv_file(args.dotenv)

    if not args.dry_run:
        api_key = get_api_key()
    else:
        api_key = ""

    data = load_dataset(args.input_json)

    contaminated_sources = load_sources(
        DEFAULT_CONTAMINATED_SOURCES if args.use_default_sources else set(),
        args.source,
        args.source_file,
    )

    if not contaminated_sources:
        raise ValueError("No contaminated sources selected. Use --source or --use-default-sources.")

    targets, selection_stats = select_targets(
        data,
        contaminated_sources=contaminated_sources,
        text_fields=args.text_field,
        max_text_chars=args.max_text_chars,
        min_text_chars=args.min_text_chars,
    )

    log.info("%d recipes selected for re-extraction", len(targets))
    log.info("Selection stats: %s", dict(selection_stats))

    done_indices = set() if args.rerun else load_checkpoint(args.checkpoint)

    if args.rerun:
        log.info("--rerun enabled: ignoring checkpoint")
    else:
        log.info("%d indices already done in checkpoint", len(done_indices))

    targets = [(index, recipe, text) for index, recipe, text in targets if index not in done_indices]

    if args.start_index is not None:
        targets = [(index, recipe, text) for index, recipe, text in targets if index >= args.start_index]
    if args.end_index is not None:
        targets = [(index, recipe, text) for index, recipe, text in targets if index <= args.end_index]
    if args.limit > 0:
        targets = targets[:args.limit]

    log.info("%d recipes planned for this run", len(targets))

    if args.dry_run:
        for index, recipe, text in targets[:args.preview]:
            log.info(
                "[dry-run] idx=%d source=%s title=%s text_chars=%d",
                index,
                recipe.get("source_id"),
                safe_str(recipe.get("title"))[:80],
                len(text),
            )
        if len(targets) > args.preview:
            log.info("[dry-run] ... %d more", len(targets) - args.preview)
        return 0

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)

    audit_rows = load_audit_rows(args.audit_csv)
    audit_indices = {int(row["index"]) for row in audit_rows if safe_str(row.get("index")).isdigit()}

    n_processed = 0
    n_failed = 0
    n_empty_text = 0
    source_counts: Counter[str] = Counter()
    started_at = time.time()

    with args.out_jsonl.open("a", encoding="utf-8") as out_fh:
        for index, recipe, text in targets:
            if index in audit_indices and not args.rerun:
                log.debug("idx=%d appears in audit CSV already; skipping", index)
                done_indices.add(index)
                continue

            if len(text) < args.min_text_chars:
                n_empty_text += 1
                done_indices.add(index)
                continue

            result = call_openrouter(
                text=text,
                api_key=api_key,
                model=args.model,
                openrouter_url=args.openrouter_url,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                request_timeout_s=args.request_timeout,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff,
                http_referer=args.http_referer,
                x_title=args.x_title,
            )

            if result is None:
                n_failed += 1
                log.error("idx=%d LLM failed; left out of checkpoint for retry", index)
                continue

            extracted = result.get("ingredients", [])
            record = build_output_record(
                index=index,
                recipe=recipe,
                extracted=extracted,
                text=text,
                model=args.model,
            )

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

            audit_rows.append(output_to_audit_row(record))
            done_indices.add(index)
            source_counts[safe_str(recipe.get("source_id"), "unknown")] += 1
            n_processed += 1

            if n_processed % args.checkpoint_every == 0:
                save_checkpoint(args.checkpoint, done_indices)
                write_audit_csv(args.audit_csv, audit_rows)

                elapsed = max(1e-9, time.time() - started_at)
                rate = n_processed / elapsed
                remaining = len(targets) - n_processed - n_failed
                eta_seconds = remaining / rate if rate else 0

                log.info(
                    "processed=%d failed=%d rate=%.3f/s ETA=%.1f min",
                    n_processed,
                    n_failed,
                    rate,
                    eta_seconds / 60,
                )

    save_checkpoint(args.checkpoint, done_indices)
    write_audit_csv(args.audit_csv, audit_rows)

    summary = {
        "inputs": {
            "input_json": str(args.input_json),
        },
        "outputs": {
            "out_jsonl": str(args.out_jsonl),
            "checkpoint": str(args.checkpoint),
            "audit_csv": str(args.audit_csv),
            "summary_json": str(args.summary_json),
        },
        "parameters": {
            "model": args.model,
            "openrouter_url": args.openrouter_url,
            "text_fields": args.text_field,
            "max_text_chars": args.max_text_chars,
            "min_text_chars": args.min_text_chars,
            "limit": args.limit,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "rerun": args.rerun,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
        },
        "contaminated_sources": sorted(contaminated_sources),
        "selection_stats": {str(key): int(value) for key, value in selection_stats.items()},
        "run_stats": {
            "planned_this_run": int(len(targets)),
            "processed": int(n_processed),
            "failed": int(n_failed),
            "empty_or_short_text": int(n_empty_text),
            "checkpoint_done_total": int(len(done_indices)),
            "processed_by_source": {str(key): int(value) for key, value in source_counts.items()},
        },
        "timestamp_utc": timestamp_utc(),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("Wrote summary JSON: %s", args.summary_json)
    log.info(
        "Done. Processed=%d Failed=%d Total checkpoint done=%d",
        n_processed,
        n_failed,
        len(done_indices),
    )

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
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"Input RELISH dataset JSON. Default: {DEFAULT_INPUT_JSON}",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=DEFAULT_OUT_JSONL,
        help=f"Output JSONL. Default: {DEFAULT_OUT_JSONL}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint JSON. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help=f"Per-recipe audit CSV. Default: {DEFAULT_AUDIT_CSV}",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"Summary JSON. Default: {DEFAULT_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="Optional .env file containing OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--openrouter-url",
        default=DEFAULT_OPENROUTER_URL,
        help="OpenRouter chat completions endpoint.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Additional contaminated source_id to process. Can be repeated.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Optional file with one contaminated source_id per line.",
    )
    parser.add_argument(
        "--no-default-sources",
        dest="use_default_sources",
        action="store_false",
        help="Do not include the built-in contaminated-source list.",
    )
    parser.set_defaults(use_default_sources=True)
    parser.add_argument(
        "--text-field",
        action="append",
        default=["recipe_text", "translation"],
        help="Recipe text field to concatenate. Can be repeated.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Maximum characters sent to the LLM per recipe. 0 = no truncation.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=10,
        help="Skip recipes with shorter concatenated text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N recipes this run. 0 = all.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Only process dataset indices >= this value.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Only process dataset indices <= this value.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore checkpoint and process selected recipes again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned recipes without API calls or output writes.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of dry-run items to preview.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_BATCH_CHECKPOINT_EVERY,
        help="Write checkpoint and audit every N processed recipes.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="OpenRouter request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per OpenRouter call.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--http-referer",
        default=DEFAULT_HTTP_REFERER,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        default=DEFAULT_X_TITLE,
        help="Optional X-Title header for OpenRouter.",
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

    if args.limit < 0:
        log.error("--limit must be >= 0")
        return 2
    if args.preview < 0:
        log.error("--preview must be >= 0")
        return 2
    if args.checkpoint_every < 1:
        log.error("--checkpoint-every must be >= 1")
        return 2
    if args.max_text_chars < 0:
        log.error("--max-text-chars must be >= 0")
        return 2
    if args.min_text_chars < 0:
        log.error("--min-text-chars must be >= 0")
        return 2
    if args.start_index is not None and args.start_index < 0:
        log.error("--start-index must be >= 0")
        return 2
    if args.end_index is not None and args.end_index < 0:
        log.error("--end-index must be >= 0")
        return 2
    if args.start_index is not None and args.end_index is not None and args.end_index < args.start_index:
        log.error("--end-index must be >= --start-index")
        return 2
    if args.request_timeout < 1:
        log.error("--request-timeout must be >= 1")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.retry_backoff < 0:
        log.error("--retry-backoff must be >= 0")
        return 2
    if args.max_output_tokens < 1:
        log.error("--max-output-tokens must be >= 1")
        return 2
    if not args.text_field:
        log.error("At least one --text-field is required")
        return 2

    try:
        return run_reextraction(args)

    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        csv.Error,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
