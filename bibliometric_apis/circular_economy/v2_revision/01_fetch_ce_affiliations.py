"""
WIDENED VERSION of 01_fetch_ce_affiliations.py

CHANGED:
  ce_affiliations.csv TO INCLUDE paper_id:
      BEFORE:  author_id, institution_id
      NOW:  author_id, paper_id, institution_id

This preserves paper–author–institution relationships for geographic attribution.
Run:
  python 01_fetch_ce_affiliations_FIXED.py \
      --corpus ./corpus_circular_economy.csv \
      --out ./ce_openalex_v2/
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


def load_existing(out: Path) -> set:
    f = out / "ce_papers_meta.csv"
    if not f.exists():
        return set()
    try:
        df = pd.read_csv(f, usecols=["doi"])
        return set(df["doi"].astype(str).str.lower())
    except Exception:
        return set()


def append_rows(path: Path, header: list, rows: list):
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="./ce_openalex_v2/")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

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

    print(f"Reading {args.corpus}...")
    df = pd.read_csv(args.corpus, usecols=["doi"])
    df["doi"] = df["doi"].apply(normalise_doi)
    df = df[df["doi"] != ""].drop_duplicates()
    all_dois = df["doi"].tolist()
    print(f"  {len(all_dois)} unique DOIs in corpus")

    done = load_existing(out)
    todo = [d for d in all_dois if d not in done]
    print(f"  {len(done)} already fetched (skipping)")
    print(f"  {len(todo)} to fetch")

    if not todo:
        print("Nothing to do.")
        return 0

    institutions_seen = set()
    if f_institutions.exists():
        try:
            inst_df = pd.read_csv(f_institutions, usecols=["openalex_id"])
            institutions_seen = set(inst_df["openalex_id"].astype(str))
        except Exception:
            pass

    headers = {"User-Agent": f"ce-review-paper/2.0 (mailto:{mailto})"}
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
                    # *** CAMBIO CLAVE: incluir wid (paper_id) ***
                    affiliations_rows.append([aid, wid, iid])
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
        # *** CAMBIO CLAVE: header con paper_id ***
        append_rows(f_affiliations,
                    ["author_id", "paper_id", "institution_id"],
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
