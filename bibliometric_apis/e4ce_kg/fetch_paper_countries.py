"""Fetch the affiliation countries of each paper from its OpenAlex authorships.

The graph links authors to institutions without the paper or the year, so the
country of a paper cannot be read from it. OpenAlex records, for every
authorship, the institutions and countries of that author in that paper. This
script requests the authorships of every paper in route_input.jsonl, keeps the
raw records, and writes one row per paper and country. Works that the ids
filter does not return (merged records) are requested one by one.

Outputs:
    paper_authorships_raw.jsonl   OpenAlex id, authorships, retrieval date
    paper_countries.csv           paper_id, country_code, authorships_with_country,
                                  authorships_total

Usage:
    python fetch_paper_countries.py

OPENALEX_API_KEY (optional) and OPENALEX_EMAIL, sent as mailto for the polite
pool, are read from the environment or from ~/Desktop/openalex/.env.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
RAW = HERE / "paper_authorships_raw.jsonl"
OUT = HERE / "paper_countries.csv"
URL = "https://api.openalex.org/works"
BATCH = 50


def short_id(paper_id):
    return paper_id.rstrip("/").split("/")[-1]


def fetch(ids, params, retries=5):
    query = {**params, "filter": "ids.openalex:" + "|".join(ids),
             "select": "id,authorships", "per-page": str(len(ids))}
    for attempt in range(retries):
        try:
            resp = requests.get(URL, params=query, timeout=60)
        except requests.RequestException as exc:
            err = str(exc)
        else:
            if resp.ok:
                return resp.json()["results"]
            if resp.status_code not in (429, 500, 502, 503, 504):
                sys.exit(f"OpenAlex HTTP {resp.status_code}: {resp.text[:300]}")
            err = f"HTTP {resp.status_code}"
        time.sleep(2 ** attempt)
    sys.exit(f"OpenAlex request failed after {retries} attempts ({err})")


def fetch_one(work_id, params, retries=5):
    """Single-work endpoint: follows OpenAlex merges that the ids filter misses."""
    for attempt in range(retries):
        try:
            resp = requests.get(f"{URL}/{work_id}", params={**params, "select": "id,authorships"},
                                timeout=60)
        except requests.RequestException as exc:
            err = str(exc)
        else:
            if resp.status_code == 404:
                return None
            if resp.ok:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503, 504):
                sys.exit(f"OpenAlex HTTP {resp.status_code}: {resp.text[:300]}")
            err = f"HTTP {resp.status_code}"
        time.sleep(2 ** attempt)
    sys.exit(f"OpenAlex request failed after {retries} attempts ({err})")


def countries(authorship):
    found = set(authorship.get("countries") or [])
    for inst in authorship.get("institutions") or []:
        if inst.get("country_code"):
            found.add(inst["country_code"])
    return found


def main():
    load_dotenv(Path.home() / "Desktop/openalex/.env")
    params = {k: os.environ[v].strip().strip('"').strip("'")
              for k, v in (("api_key", "OPENALEX_API_KEY"), ("mailto", "OPENALEX_EMAIL"))
              if os.environ.get(v)}
    papers = [json.loads(line)["paper_id"] for line in
              (HERE / "route_input.jsonl").open(encoding="utf-8")]
    ids = sorted({short_id(p) for p in papers if "openalex.org/W" in p})
    skipped = len(papers) - len(ids)

    done = {}
    if RAW.exists():
        for line in RAW.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["id"]] = rec
    todo = [i for i in ids if i not in done]
    print(f"papers {len(papers)} | OpenAlex ids {len(ids)} | without OpenAlex id {skipped} | "
          f"already fetched {len(ids) - len(todo)} | to fetch {len(todo)}")

    with RAW.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for work in fetch(batch, params):
                rec = {"id": short_id(work["id"]), "authorships": work.get("authorships") or [],
                       "retrieved_at": stamp}
                done[rec["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (start // BATCH) % 20 == 0:
                print(f"  {min(start + BATCH, len(todo))}/{len(todo)}", flush=True)

        missing = [i for i in ids if i not in done]
        if missing:
            print(f"  {len(missing)} ids not returned by the ids filter; requesting one by one")
        for work_id in missing:
            work = fetch_one(work_id, params)
            if work is None:
                continue
            rec = {"id": work_id, "canonical_id": short_id(work["id"]),
                   "authorships": work.get("authorships") or [],
                   "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            done[work_id] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    missing = [i for i in ids if i not in done]
    rows, no_country = [], 0
    for pid in papers:
        rec = done.get(short_id(pid))
        if rec is None:
            continue
        per_author = [countries(a) for a in rec["authorships"]]
        found = sorted(set().union(*per_author)) if per_author else []
        if not found:
            no_country += 1
        for cc in found:
            rows.append([pid, cc, sum(cc in c for c in per_author), len(per_author)])
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_id", "country_code", "authorships_with_country", "authorships_total"])
        w.writerows(rows)
    print(f"wrote {OUT.name}: {len(rows)} rows | papers not returned by OpenAlex {len(missing)} | "
          f"papers without any country {no_country}")


if __name__ == "__main__":
    main()
