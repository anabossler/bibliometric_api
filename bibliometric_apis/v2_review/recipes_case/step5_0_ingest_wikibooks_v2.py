"""
==============================

Ingest contemporary European recipes from Wikibooks Cookbook.

This script fetches selected Wikibooks cuisine index pages, extracts linked
Cookbook recipe pages, parses ingredient sections, and emits records in the
RELISH JSON shape used by the historical recipe pipeline.

Improvements over the first Wikibooks ingest:

  - broader ingredient-header detection
  - fallback bullet-list heuristic when no ingredient header is found
  - wiki-table ingredient parsing
  - resumable API cache
  - diagnosis of rejected pages
  - configurable API, output paths, retry policy, sleep, and user-agent

Default input
-------------

  Wikibooks API, no local input file.

Default outputs
---------------

  data/wikibooks_contemporary.json
  data/wikibooks_cache.jsonl
  data/wikibooks_ingest_stats.json
  data/wikibooks_rejections.csv          when --diagnose or --write-rejections

Usage
-----

  python step5_0_ingest_wikibooks_v2.py --diagnose

  python step5_0_ingest_wikibooks_v2.py --dry-run

  python step5_0_ingest_wikibooks_v2.py --sleep 2

  python step5_0_ingest_wikibooks_v2.py --cuisines spain france italy --sleep 2

  python step5_0_ingest_wikibooks_v2.py --max-per-cuisine 50 --refresh-cache

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step5_0_ingest_wikibooks_v2")


DEFAULT_API_BASE = "https://en.wikibooks.org/w/api.php"
DEFAULT_USER_AGENT = "relish-recipes-research/1.0 (academic; contact: project-local)"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "wikibooks_contemporary.json"
DEFAULT_CACHE = DEFAULT_DATA_DIR / "wikibooks_cache.jsonl"
DEFAULT_STATS = DEFAULT_DATA_DIR / "wikibooks_ingest_stats.json"
DEFAULT_REJECTIONS = DEFAULT_DATA_DIR / "wikibooks_rejections.csv"
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_SOURCE_YEAR = 2024


# ---------------------------------------------------------------------------
# European cuisine index pages
# ---------------------------------------------------------------------------

EUROPEAN_INDEX_PAGES: dict[str, tuple[str, str]] = {
    "spain": ("Cookbook:Cuisine of Spain", "Spain"),
    "france": ("Cookbook:Cuisine of France", "France"),
    "italy": ("Cookbook:Cuisine of Italy", "Italy"),
    "germany": ("Cookbook:Cuisine of Germany", "Germany"),
    "portugal": ("Cookbook:Cuisine of Portugal", "Portugal"),
    "greece": ("Cookbook:Cuisine of Greece", "Greece"),
    "uk": ("Cookbook:Cuisine of the United Kingdom", "United Kingdom"),
    "ireland": ("Cookbook:Cuisine of Ireland", "Ireland"),
    "sweden": ("Cookbook:Cuisine of Sweden", "Sweden"),
    "norway": ("Cookbook:Cuisine of Norway", "Norway"),
    "denmark": ("Cookbook:Cuisine of Denmark", "Denmark"),
    "finland": ("Cookbook:Cuisine of Finland", "Finland"),
    "poland": ("Cookbook:Cuisine of Poland", "Poland"),
    "hungary": ("Cookbook:Cuisine of Hungary", "Hungary"),
    "austria": ("Cookbook:Cuisine of Austria", "Austria"),
    "switzerland": ("Cookbook:Cuisine of Switzerland", "Switzerland"),
    "belgium": ("Cookbook:Cuisine of Belgium", "Belgium"),
    "netherlands": ("Cookbook:Cuisine of the Netherlands", "Netherlands"),
    "russia": ("Cookbook:Cuisine of Russia", "Russia"),
    "romania": ("Cookbook:Cuisine of Romania", "Romania"),
    "ukraine": ("Cookbook:Cuisine of Ukraine", "Ukraine"),
    "czech": ("Cookbook:Czech cuisine", "Czech Republic"),
    "catalonia": ("Cookbook:Catalan cuisine", "Catalonia"),
}


# ---------------------------------------------------------------------------
# Wiki and ingredient parsing regexes
# ---------------------------------------------------------------------------

QTY_RE = re.compile(
    r"^[\d\s/.½¼¾⅓⅔⅛⅜⅝⅞~≈×x–—-]*\s*"
    r"(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|ounces?|oz\.?|pounds?|lbs?\.?|"
    r"grams?|g|kg|millilit(?:er|re)s?|ml|lit(?:er|re)s?|l|quarts?|pints?|gallons?|"
    r"cans?|packages?|pkg\.?|sticks?|slices?|pieces?|heads?|cloves?|"
    r"bunch(?:es)?|sprigs?|pinch(?:es)?|dash(?:es)?|handfuls?|drops?|"
    r"small|medium|large|inch(?:es)?|cm|dl|cl)\b\.?\s*(?:of\s+)?",
    re.IGNORECASE,
)

PREP_SUFFIXES = re.compile(
    r",?\s*(?:"
    r"finely|coarsely|roughly|thinly|freshly|"
    r"chopped|diced|minced|sliced|grated|shredded|crushed|ground|"
    r"melted|softened|divided|optional|to taste|"
    r"at room temperature|room temperature|peeled|seeded|cored|trimmed|"
    r"drained|rinsed|cooked|uncooked|frozen|thawed|packed|sifted|"
    r"toasted|dried|fresh|cut into.*|about.*|approximately.*|"
    r"or more|or less|as needed|cleaned|washed|deveined|deboned|"
    r"beaten|whisked|warmed?|chilled|cold|hot|lukewarm"
    r")\s*$",
    re.IGNORECASE,
)

PARENS_RE = re.compile(r"\([^)]*\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
WIKI_TEMPLATE_RE = re.compile(r"\{\{(?:[^{}]|\{[^{}]*\})*}}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/>]*/>", re.IGNORECASE | re.DOTALL)
BULLET_RE = re.compile(r"^\s*[*#]+\s*(.+?)\s*$")
HEADER_RE = re.compile(r"(?m)^==+\s*(.+?)\s*==+\s*$")


INGREDIENTS_HEADER_RE = re.compile(
    r"^(?:"
    r"ingredients?"
    r"|materials?"
    r"|what you(?:'ll)? need"
    r"|you will need"
    r"|things you(?:'ll)? need"
    r"|shopping list"
    r"|required items"
    r"|supplies"
    r"|for the [\w\s\-]{2,40}"
    r"|zutaten"
    r"|ingr[eé]dients?"
    r"|ingredientes?"
    r"|ingredienti"
    r")$",
    re.IGNORECASE,
)

PROCEDURE_HEADER_RE = re.compile(
    r"^(?:"
    r"procedure|method|directions?|instructions?|preparation|steps?"
    r"|how to make|cooking|preparaci[oó]n|preparazione"
    r")$",
    re.IGNORECASE,
)

FOOD_SIGNAL_RE = re.compile(
    r"\b(?:cup|tbsp|tsp|tablespoon|teaspoon|gram|ounce|oz|pound|lb|"
    r"kg|ml|lit(?:er|re)|pinch|dash|clove|slice|piece|"
    r"salt|pepper|sugar|butter|oil|flour|egg|milk|cream|water|"
    r"onion|garlic|cheese|chicken|beef|pork|fish|lemon|"
    r"tomato|potato|rice|bread|wine|vinegar|sauce|stock|broth|"
    r"½|¼|¾|⅓|⅔)\b",
    re.IGNORECASE,
)


NON_INGREDIENT_PHRASES = {
    "method",
    "procedure",
    "direction",
    "directions",
    "instructions",
    "step",
    "preheat",
    "serve",
    "serves",
    "note",
    "tip",
    "see also",
    "external links",
    "references",
    "category",
}


# ---------------------------------------------------------------------------
# Markup and ingredient cleaning
# ---------------------------------------------------------------------------

def clean_wiki_markup(text: str) -> str:
    """Strip common wiki markup while preserving readable text."""
    text = str(text or "")
    text = REF_RE.sub("", text)

    def replace_link(match: re.Match[str]) -> str:
        inner = match.group(1)
        target, _, label = inner.partition("|")
        if label:
            return label.strip()
        target = target.strip()
        if target.startswith("Cookbook:"):
            return target.split(":", 1)[1]
        return target.split("#", 1)[0].strip()

    previous = None
    while previous != text:
        previous = text
        text = WIKI_LINK_RE.sub(replace_link, text)
        text = WIKI_TEMPLATE_RE.sub("", text)

    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_ingredient_name(raw: str) -> str:
    """Extract a normalized ingredient name from a raw Wikibooks ingredient line."""
    text = clean_wiki_markup(raw).lower()
    text = PARENS_RE.sub("", text).strip()
    text = re.sub(r"^[:;,\-–—*#|]+", "", text).strip()
    text = QTY_RE.sub("", text).strip()
    text = PREP_SUFFIXES.sub("", text).strip()

    # Drop leading quantity after first pass, e.g. "2 large eggs" -> "eggs".
    text = re.sub(r"^\d+(?:[./]\d+)?\s+", "", text)
    text = re.sub(r"\b(?:small|medium|large)\b\s+", "", text, count=1)

    text = text.strip(" ,.-;:*#|")
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_instruction(name: str) -> bool:
    normalized = name.lower().strip()
    if not normalized:
        return True

    if any(phrase in normalized for phrase in NON_INGREDIENT_PHRASES):
        return True

    if len(normalized.split()) > 8:
        return True

    if re.search(r"\b(?:preheat|bake|boil|fry|simmer|stir|mix|combine|serve)\b", normalized):
        return True

    return False


def ingredients_to_relish(raw_list: list[str], *, max_raw_chars: int = 200) -> list[dict[str, Any]]:
    """Convert raw ingredient lines into RELISH ingredient dictionaries."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in raw_list:
        name = parse_ingredient_name(raw)
        if not name or len(name) < 2 or name in seen:
            continue
        if looks_like_instruction(name):
            continue

        seen.add(name)
        result.append({
            "name": name,
            "confidence_score": 0.9,
            "specific_forms": [str(raw).strip()[:max_raw_chars]],
        })

    return result


# ---------------------------------------------------------------------------
# Cache and API
# ---------------------------------------------------------------------------

def cache_key(params: dict[str, Any]) -> str:
    relevant = dict(params)
    relevant.setdefault("format", "json")
    relevant.setdefault("formatversion", 2)
    return json.dumps(sorted(relevant.items()), ensure_ascii=False)


def load_cache(path: Path, *, enabled: bool) -> dict[str, dict[str, Any]]:
    if not enabled or not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                params = row.get("params")
                response = row.get("response")
                if isinstance(params, dict) and isinstance(response, dict):
                    cache[cache_key(params)] = response
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping malformed cache line %d in %s", line_no, path)

    log.info("Loaded %d cached Wikibooks API responses", len(cache))
    return cache


def append_cache(path: Path, *, params: dict[str, Any], response: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"params": params, "response": response}

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def api_get(
    *,
    params: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    api_base: str,
    sleep_s: float,
    timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    user_agent: str,
    use_cache: bool,
    write_cache: bool,
    refresh_cache: bool,
) -> dict[str, Any]:
    params_full = {**params, "format": "json", "formatversion": 2}
    key = cache_key(params_full)

    if use_cache and not refresh_cache and key in cache:
        return cache[key]

    url = f"{api_base}?{urlencode(params_full)}"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))

            cache[key] = data

            if write_cache:
                append_cache(cache_path, params=params_full, response=data)

            if sleep_s > 0:
                time.sleep(sleep_s)

            return data

        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                log.warning("Non-retryable HTTP error for %s: %s", params.get("page", "?"), exc.code)
                return {}

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "HTTP error for %s attempt %d/%d: %s. Sleeping %.1fs",
                params.get("page", "?"),
                attempt + 1,
                max_retries,
                exc.code,
                wait,
            )
            time.sleep(wait)

        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "API error for %s attempt %d/%d: %s. Sleeping %.1fs",
                params.get("page", "?"),
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    log.warning("API call failed for %s after %d retries: %s", params.get("page", "?"), max_retries, last_error)
    return {}


def fetch_page_wikitext(title: str, api_args: dict[str, Any]) -> str:
    data = api_get(
        params={"action": "parse", "page": title, "prop": "wikitext", "redirects": 1},
        **api_args,
    )
    parsed = data.get("parse") or {}
    wikitext = parsed.get("wikitext")

    if isinstance(wikitext, dict):
        wikitext = wikitext.get("*", "")

    return wikitext if isinstance(wikitext, str) else ""


# ---------------------------------------------------------------------------
# Page discovery and extraction
# ---------------------------------------------------------------------------

def extract_cookbook_links(wikitext: str) -> list[str]:
    """Extract candidate Cookbook recipe-page links from an index page."""
    titles: set[str] = set()

    for match in WIKI_LINK_RE.finditer(wikitext or ""):
        inner = match.group(1)
        target = inner.split("|", 1)[0].strip()
        target = target.split("#", 1)[0].strip()

        if not target.startswith("Cookbook:") or target == "Cookbook:":
            continue

        tail = target[len("Cookbook:"):].strip()

        if not tail:
            continue
        if tail.startswith(("Cuisine of", "Cuisines", "Recipes ", "Table of", "Category:")):
            continue
        if tail in {"Recipes", "Cuisines", "Glossary", "Ingredients"}:
            continue

        titles.add(target)

    return sorted(titles)


def extract_bullets(text: str) -> list[str]:
    items: list[str] = []

    for line in text.splitlines():
        match = BULLET_RE.match(line)
        if not match:
            continue

        cleaned = clean_wiki_markup(match.group(1))
        if cleaned and len(cleaned) > 1:
            items.append(cleaned)

    return items


def extract_table_ingredients(text: str) -> list[str]:
    """Extract ingredient-like items from wiki table markup."""
    items: list[str] = []
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("{|"):
            in_table = True
            continue
        if stripped.startswith("|}"):
            in_table = False
            continue
        if not in_table:
            continue
        if not stripped.startswith("|") or stripped.startswith("|-"):
            continue

        cells = re.split(r"\|\|", stripped.lstrip("| "))
        for cell in cells:
            cleaned = clean_wiki_markup(cell).strip()
            if cleaned and len(cleaned) > 1 and re.search(r"[A-Za-z]", cleaned):
                items.append(cleaned)

    return items


def heading_spans(wikitext: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(1).strip()) for match in HEADER_RE.finditer(wikitext)]


def section_after_heading(wikitext: str, heading_end: int, all_headings: list[tuple[int, int, str]]) -> str:
    next_starts = [start for start, _, _ in all_headings if start > heading_end]
    end = min(next_starts) if next_starts else len(wikitext)
    return wikitext[heading_end:end]


def is_ingredient_heading(title: str) -> bool:
    title = clean_wiki_markup(title).strip()
    return bool(INGREDIENTS_HEADER_RE.match(title))


def extract_all_ingredient_sections(wikitext: str) -> list[str]:
    """Extract ingredient items using headers, table parsing, and fallback heuristic."""
    all_items: list[str] = []
    headings = heading_spans(wikitext)

    for _, heading_end, heading_title in headings:
        if not is_ingredient_heading(heading_title):
            continue

        section = section_after_heading(wikitext, heading_end, headings)
        bullets = extract_bullets(section)
        tables = extract_table_ingredients(section)

        if bullets:
            all_items.extend(bullets)
        if tables:
            all_items.extend(tables)

    if not all_items:
        all_items = fallback_bullet_heuristic(wikitext)

    # Preserve order while removing exact duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for item in all_items:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def fallback_bullet_heuristic(wikitext: str) -> list[str]:
    """Find the largest contiguous ingredient-like bullet block."""
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in wikitext.splitlines():
        match = BULLET_RE.match(line)
        if match:
            cleaned = clean_wiki_markup(match.group(1))
            if cleaned and len(cleaned) > 1:
                current.append(cleaned)
            continue

        if current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    best_block: list[str] = []
    best_score = 0

    for block in blocks:
        if len(block) < 3:
            continue

        score = sum(1 for item in block if FOOD_SIGNAL_RE.search(item))
        if score > best_score and score >= len(block) * 0.4:
            best_block = block
            best_score = score

    return best_block


def is_recipe_page(wikitext: str) -> tuple[bool, str]:
    """Return (is_recipe, rejection_reason)."""
    if not wikitext:
        return False, "empty_page"
    if len(wikitext) < 100:
        return False, "too_short"

    headings = [title for _, _, title in heading_spans(wikitext)]

    if any(is_ingredient_heading(title) for title in headings):
        return True, ""

    if any(PROCEDURE_HEADER_RE.match(clean_wiki_markup(title).strip()) for title in headings):
        if len(extract_bullets(wikitext)) >= 3:
            return True, ""

    fallback = fallback_bullet_heuristic(wikitext)
    if len(fallback) >= 3:
        return True, ""

    if len(wikitext) < 200:
        return False, "too_short"

    if not headings:
        return False, "no_headers"

    return False, f"unrecognized_headers:{'; '.join(headings[:5])}"


def clean_title(title: str) -> str:
    return title.split(":", 1)[-1].replace("_", " ").strip()


def recipe_url(title: str) -> str:
    return f"https://en.wikibooks.org/wiki/{quote(title.replace(' ', '_'), safe=':/')}"


def record_id(cuisine_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", clean_title(title).lower()).strip("_")
    return f"wikibooks_en_{cuisine_id}::{slug}"


# ---------------------------------------------------------------------------
# Cuisine processing
# ---------------------------------------------------------------------------

def process_cuisine(
    *,
    cuisine_id: str,
    index_page: str,
    place: str,
    max_recipes: int,
    source_year: int,
    api_args: dict[str, Any],
    diagnose: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    log.info("[%s] Fetching index: %s", cuisine_id, index_page)
    index_text = fetch_page_wikitext(index_page, api_args)

    if not index_text:
        log.warning("[%s] Index page empty or not found", cuisine_id)
        return [], {"index_missing": 1}, []

    recipe_titles = extract_cookbook_links(index_text)
    log.info("[%s] %d candidate recipe links", cuisine_id, len(recipe_titles))

    if max_recipes > 0:
        recipe_titles = recipe_titles[:max_recipes]

    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    rejections: list[dict[str, Any]] = []

    for index, title in enumerate(recipe_titles, start=1):
        wikitext = fetch_page_wikitext(title, api_args)
        is_recipe, reason = is_recipe_page(wikitext)

        if not is_recipe:
            stats["not_recipe"] += 1
            stats[f"reject:{reason}"] += 1
            if diagnose:
                rejections.append({
                    "cuisine_id": cuisine_id,
                    "title": clean_title(title),
                    "wikibooks_page": title,
                    "reason": reason,
                    "length": len(wikitext) if wikitext else 0,
                    "raw_count": 0,
                    "raw_sample": "",
                })
            continue

        raw_ingredients = extract_all_ingredient_sections(wikitext)
        ingredients = ingredients_to_relish(raw_ingredients)

        if not ingredients:
            stats["no_ingredients_parsed"] += 1
            rejections.append({
                "cuisine_id": cuisine_id,
                "title": clean_title(title),
                "wikibooks_page": title,
                "reason": "ingredients_parsed_to_empty",
                "length": len(wikitext) if wikitext else 0,
                "raw_count": len(raw_ingredients),
                "raw_sample": json.dumps(raw_ingredients[:5], ensure_ascii=False),
            })
            continue

        clean_text = clean_wiki_markup(wikitext)
        title_clean = clean_title(title)

        records.append({
            "source_id": f"wikibooks_en_{cuisine_id}",
            "source_title": "Wikibooks Cookbook",
            "source_author": "Wikibooks contributors",
            "source_year": source_year,
            "source_place": place,
            "source_language": "en",
            "title": title_clean,
            "recipe_id": record_id(cuisine_id, title),
            "recipe_text": clean_text[:5000],
            "translation": clean_text[:5000],
            "ingredients": ingredients,
            "tools": [],
            "actions": [],
            "wikibooks_page": title,
            "wikibooks_url": recipe_url(title),
            "ingest_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        stats["recipes_kept"] += 1

        if index % 20 == 0:
            log.info(
                "[%s] %d/%d processed (%d kept)",
                cuisine_id,
                index,
                len(recipe_titles),
                stats["recipes_kept"],
            )

    stats["candidate_links"] = len(recipe_titles)

    log.info(
        "[%s] Done: kept %d/%d candidates",
        cuisine_id,
        stats["recipes_kept"],
        len(recipe_titles),
    )

    if diagnose and rejections:
        log.info("[%s] Rejections:", cuisine_id)
        for item in rejections[:50]:
            log.info("  %-40s reason=%s", item["title"][:40], item["reason"])
        if len(rejections) > 50:
            log.info("  ... %d more rejections", len(rejections) - 50)

    return records, dict(stats), rejections


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    log.info("Wrote JSON: %s", path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cuisines",
        nargs="*",
        default=None,
        choices=sorted(EUROPEAN_INDEX_PAGES.keys()),
        help="Specific cuisines to fetch. Default: all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output RELISH JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="API cache JSONL. Default: data-dir/wikibooks_cache.jsonl.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Stats JSON. Default: data-dir/wikibooks_ingest_stats.json.",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        help="Rejections CSV. Default: data-dir/wikibooks_rejections.csv.",
    )
    parser.add_argument(
        "--max-per-cuisine",
        type=int,
        default=0,
        help="Maximum recipe links per cuisine. 0 = unlimited.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds between uncached API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per API call.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Wikibooks API endpoint.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header.",
    )
    parser.add_argument(
        "--source-year",
        type=int,
        default=DEFAULT_SOURCE_YEAR,
        help="source_year value written to records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate recipe links only; do not fetch recipe pages.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show and write detailed rejection reasons.",
    )
    parser.add_argument(
        "--write-rejections",
        action="store_true",
        help="Write rejections CSV even when --diagnose is not enabled.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read existing cache.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Do not append new API responses to cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached responses and re-fetch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    args.cache = args.cache or args.data_dir / "wikibooks_cache.jsonl"
    args.stats = args.stats or args.data_dir / "wikibooks_ingest_stats.json"
    args.rejections = args.rejections or args.data_dir / "wikibooks_rejections.csv"

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.max_per_cuisine < 0:
        log.error("--max-per-cuisine must be >= 0")
        return 2
    if args.sleep < 0:
        log.error("--sleep must be >= 0")
        return 2
    if args.timeout < 1:
        log.error("--timeout must be >= 1")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.retry_backoff < 0:
        log.error("--retry-backoff must be >= 0")
        return 2

    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)

        cache = load_cache(args.cache, enabled=not args.no_cache and not args.refresh_cache)
        cuisines_to_fetch = args.cuisines or sorted(EUROPEAN_INDEX_PAGES.keys())

        log.info(
            "Will fetch %d cuisines: %s",
            len(cuisines_to_fetch),
            ", ".join(cuisines_to_fetch),
        )

        api_args = {
            "cache": cache,
            "cache_path": args.cache,
            "api_base": args.api_base,
            "sleep_s": args.sleep,
            "timeout_s": args.timeout,
            "max_retries": args.max_retries,
            "retry_backoff_s": args.retry_backoff,
            "user_agent": args.user_agent,
            "use_cache": not args.no_cache,
            "write_cache": not args.no_write_cache,
            "refresh_cache": args.refresh_cache,
        }

        all_records: list[dict[str, Any]] = []
        all_stats: dict[str, Any] = {}
        all_rejections: list[dict[str, Any]] = []

        for cuisine_id in cuisines_to_fetch:
            index_page, place = EUROPEAN_INDEX_PAGES[cuisine_id]

            if args.dry_run:
                index_text = fetch_page_wikitext(index_page, api_args)
                links = extract_cookbook_links(index_text)
                all_stats[cuisine_id] = {"candidate_links": len(links)}
                log.info("[dry-run] %s: %d candidate recipe links", cuisine_id, len(links))
                continue

            records, stats, rejections = process_cuisine(
                cuisine_id=cuisine_id,
                index_page=index_page,
                place=place,
                max_recipes=args.max_per_cuisine,
                source_year=args.source_year,
                api_args=api_args,
                diagnose=args.diagnose,
            )

            all_records.extend(records)
            all_stats[cuisine_id] = stats
            all_rejections.extend(rejections)

        stats_payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "cuisines": cuisines_to_fetch,
                "max_per_cuisine": args.max_per_cuisine,
                "sleep": args.sleep,
                "timeout": args.timeout,
                "source_year": args.source_year,
                "api_base": args.api_base,
                "no_cache": args.no_cache,
                "refresh_cache": args.refresh_cache,
                "write_cache": not args.no_write_cache,
                "dry_run": args.dry_run,
            },
            "total_records": len(all_records),
            "per_cuisine": all_stats,
            "total_rejections_recorded": len(all_rejections),
        }

        if args.dry_run:
            total_links = sum(item.get("candidate_links", 0) for item in all_stats.values())
            log.info("=" * 72)
            log.info("DRY RUN SUMMARY")
            log.info("Total candidate recipe links: %d", total_links)
            for cuisine_id, stats in all_stats.items():
                log.info("  %-15s %d", cuisine_id, stats.get("candidate_links", 0))
            write_json(args.stats, stats_payload)
            return 0

        write_json(args.output, all_records, indent=1)
        write_json(args.stats, stats_payload)

        if args.diagnose or args.write_rejections:
            write_csv(args.rejections, all_rejections)

        log.info("=" * 72)
        log.info("INGEST SUMMARY")
        log.info("Total contemporary recipes: %d", len(all_records))
        for cuisine_id, stats in all_stats.items():
            log.info(
                "  %-15s kept=%-3d rejected=%-3d no_parse=%-3d candidates=%-3d",
                cuisine_id,
                stats.get("recipes_kept", 0),
                stats.get("not_recipe", 0),
                stats.get("no_ingredients_parsed", 0),
                stats.get("candidate_links", 0),
            )

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        HTTPError,
        URLError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
