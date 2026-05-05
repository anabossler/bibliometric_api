"""
Fetches author affiliations from the OpenAlex API for the circular economy
corpus, producing CSVs compatible with ce_geography.py:
    ce_authorships.csv   (author_id, paper_id, position)
    ce_affiliations.csv  (author_id, institution_id)
    ce_papers_meta.csv   (openalex_id, doi, title, year, citations, type, is_oa)
    ce_institutions.csv  (openalex_id, name, country, type)

The script:
  - Reads DOIs from corpus_circular_economy.csv
  - Reads OPENALEX_EMAIL and OPENALEX_SLEEP from .env (no argument needed)
  - Queries OpenAlex /works in batches of 50 via filter doi:doi1|doi2|...
  - Polite pool: uses OPENALEX_EMAIL for rate-limited 10 req/s access
  - Resumable: skips DOIs already present in --out/ce_papers_meta.csv
  - Writes to disk every batch (no data loss on interrupt)

Run:
  python fetch_ce_affiliations.py \\
      --corpus ./corpus_circular_economy.csv \\
      --out ./ce_openalex/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


OPENALEX_BASE = "https://api.openalex.org/works"


def load_env(env_path: Path = Path(".env")) -> dict:
    """Minimal .env reader (no external deps). Returns dict of KEY=VALUE pairs."""
    env = {}
    if not env_path.exists():
        return env
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def normalise_doi(d: str) -> str:
    if not isinstance(d, str):
        return ""
    d = d.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_existing(out: Path) -> set[str]:
    """Return set of DOIs already fetched (for resumability)."""
    f = out / "ce_papers_meta.csv"
    if not f.exists():
        return set()
    try:
        df = pd.read_csv(f, usecols=["doi"])
        return set(df["doi"].astype(str).str.lower())
    except Exception:
        return set()


def append_rows(path: Path, header: list[str], rows: list[list]):
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="CSV with a 'doi' column (your corpus_circular_economy.csv)")
    ap.add_argument("--out", default="./ce_openalex/", help="Output directory")
    ap.add_argument("--env", default=".env", help="Path to .env file")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="DOIs per request (max ~50)")
    ap.add_argument("--sleep", type=float, default=None,
                    help="Override OPENALEX_SLEEP from .env")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    # Load .env
    env = load_env(Path(args.env))
    mailto = env.get("OPENALEX_EMAIL")
    if not mailto:
        print(f"ERROR: OPENALEX_EMAIL not found in {args.env}", file=sys.stderr)
        return 1
    sleep_s = args.sleep if args.sleep is not None else float(env.get("OPENALEX_SLEEP", "0.2"))
    print(f"Using mailto={mailto}, sleep={sleep_s}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    f_papers = out / "ce_papers_meta.csv"
    f_authorships = out / "ce_authorships.csv"
    f_affiliations = out / "ce_affiliations.csv"
    f_institutions = out / "ce_institutions.csv"
    f_failed = out / "ce_failed_dois.csv"

    # Load DOIs
    print(f"Reading {args.corpus}...")
    df = pd.read_csv(args.corpus, usecols=["doi"])
    df["doi"] = df["doi"].apply(normalise_doi)
    df = df[df["doi"] != ""].drop_duplicates()
    all_dois = df["doi"].tolist()
    print(f"  {len(all_dois)} unique DOIs in corpus")

    # Resumability
    done = load_existing(out)
    todo = [d for d in all_dois if d not in done]
    print(f"  {len(done)} already fetched (skipping)")
    print(f"  {len(todo)} to fetch")

    if not todo:
        print("Nothing to do.")
        return 0

    # Track institutions already written so we don't duplicate
    institutions_seen: set[str] = set()
    if f_institutions.exists():
        try:
            inst_df = pd.read_csv(f_institutions, usecols=["openalex_id"])
            institutions_seen = set(inst_df["openalex_id"].astype(str))
        except Exception:
            pass

    headers = {"User-Agent": f"ce-review-paper/1.0 (mailto:{mailto})"}
    session = requests.Session()
    session.headers.update(headers)

    n_papers_fetched = 0
    n_batches = 0
    t_start = time.time()

    for batch in chunked(todo, args.batch_size):
        n_batches += 1
        doi_filter = "doi:" + "|".join(batch)
        params = {
            "filter": doi_filter,
            "per-page": args.batch_size,
            "mailto": mailto,
            "select": "id,doi,title,publication_year,cited_by_count,type,open_access,authorships",
        }

        last_err = None
        for attempt in range(args.max_retries):
            try:
                r = session.get(OPENALEX_BASE, params=params, timeout=60)
                if r.status_code == 200:
                    break
                last_err = f"HTTP {r.status_code}"
                time.sleep(2 ** attempt)
            except Exception as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        else:
            print(f"  batch {n_batches}: FAILED after retries ({last_err})")
            append_rows(f_failed, ["doi", "reason"],
                        [[d, last_err or "unknown"] for d in batch])
            time.sleep(sleep_s)
            continue

        try:
            data = r.json()
        except Exception as e:
            print(f"  batch {n_batches}: bad JSON ({e})")
            append_rows(f_failed, ["doi", "reason"],
                        [[d, f"bad-json: {e}"] for d in batch])
            time.sleep(sleep_s)
            continue

        results = data.get("results", [])

        papers_rows, authorships_rows = [], []
        affiliations_rows, institutions_rows = [], []
        returned_dois = set()

        for w in results:
            wid = w.get("id", "")
            doi_full = w.get("doi", "") or ""
            doi_norm = normalise_doi(doi_full)
            returned_dois.add(doi_norm)

            papers_rows.append([
                wid,
                doi_norm,
                (w.get("title") or "").replace("\n", " "),
                w.get("publication_year"),
                w.get("cited_by_count"),
                w.get("type") or "",
                (w.get("open_access") or {}).get("is_oa", False),
            ])

            for a in w.get("authorships", []):
                author = a.get("author") or {}
                aid = author.get("id", "")
                if not aid:
                    continue
                authorships_rows.append([
                    aid, wid, a.get("author_position") or "",
                ])
                for inst in a.get("institutions") or []:
                    iid = inst.get("id", "")
                    if not iid:
                        continue
                    affiliations_rows.append([aid, iid])
                    if iid not in institutions_seen:
                        institutions_seen.add(iid)
                        institutions_rows.append([
                            iid,
                            inst.get("display_name") or "",
                            inst.get("country_code") or "",
                            inst.get("type") or "",
                        ])

        missing = [d for d in batch if d not in returned_dois]
        if missing:
            append_rows(f_failed, ["doi", "reason"],
                        [[d, "not-found"] for d in missing])

        append_rows(f_papers,
                    ["openalex_id", "doi", "title", "year",
                     "citations", "type", "is_oa"],
                    papers_rows)
        append_rows(f_authorships,
                    ["author_id", "paper_id", "position"],
                    authorships_rows)
        append_rows(f_affiliations,
                    ["author_id", "institution_id"],
                    affiliations_rows)
        if institutions_rows:
            append_rows(f_institutions,
                        ["openalex_id", "name", "country", "type"],
                        institutions_rows)

        n_papers_fetched += len(results)

        if n_batches % 10 == 0:
            elapsed = time.time() - t_start
            rate = n_papers_fetched / max(elapsed, 1)
            print(f"  batch {n_batches}: {n_papers_fetched} papers fetched"
                  f" | {rate:.1f} papers/s | elapsed {elapsed:.0f}s")

        time.sleep(sleep_s)

    elapsed = time.time() - t_start
    print(f"\nDONE. Fetched {n_papers_fetched} papers in {elapsed:.0f}s.")
    print(f"Output in {out}/")
    print("Files:")
    for f in [f_papers, f_authorships, f_affiliations, f_institutions, f_failed]:
        if f.exists():
            print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
