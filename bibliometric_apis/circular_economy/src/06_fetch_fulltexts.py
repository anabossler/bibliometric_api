"""

Fetches up to 50 full-text papers per cluster from the CE corpus,
building on the same OpenAlex pipeline as fetch_ce_affiliations.py.

Strategy (tried in order for each paper):
  1. OpenAlex best_oa_location → PDF URL → download & extract text via pdfminer
  2. Unpaywall API → PDF URL → same
  3. Semantic Scholar /paper/{doi} → open_access_pdf → same
  4. OpenAlex abstract_inverted_index → reconstruct abstract (fallback)

Output structure:
  --out/
    cluster_C2/
      <doi_slug>.txt       (full text or abstract fallback)
      <doi_slug>.meta.json (doi, title, year, cluster, source_used)
    cluster_C3/
      ...
    fetch_summary.csv      (one row per paper: doi, cluster, source, chars)

Inputs required:
  --corpus    corpus_circular_economy.csv   must have columns: doi, cluster
  --env       .env file with OPENALEX_EMAIL (and optionally UNPAYWALL_EMAIL)

Usage:
  python fetch_cluster_fulltexts.py \\
      --corpus ./corpus_circular_economy.csv \\
      --out    ./cluster_fulltexts/ \\
      --n      50 \\
      --clusters C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C14

Dependencies:
  pip install requests pandas pdfminer.six tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Optional: pdfminer for PDF → text
# ---------------------------------------------------------------------------
try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    PDFMINER_OK = True
except ImportError:
    PDFMINER_OK = False
    print("WARNING: pdfminer.six not installed — PDF extraction disabled. "
          "Install with: pip install pdfminer.six", file=sys.stderr)

OPENALEX_BASE = "https://api.openalex.org/works"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
S2_BASE        = "https://api.semanticscholar.org/graph/v1/paper"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_env(env_path: Path) -> dict:
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


def doi_to_slug(doi: str) -> str:
    """Convert DOI to a safe filename."""
    return re.sub(r"[^\w\-]", "_", doi)[:120]


def reconstruct_abstract(inv_index: dict) -> str:
    """Reconstruct abstract from OpenAlex abstract_inverted_index."""
    if not inv_index:
        return ""
    positions = {}
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfminer."""
    if not PDFMINER_OK:
        return ""
    try:
        out = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            out,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        return out.getvalue().strip()
    except Exception as e:
        return ""


def download_pdf(url: str, session: requests.Session, timeout: int = 30) -> Optional[bytes]:
    """Download a PDF, return bytes or None."""
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and ("pdf" in ct or url.lower().endswith(".pdf")):
            return r.content
        # Sometimes servers lie about content-type; check magic bytes
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            return r.content
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Source strategies
# ---------------------------------------------------------------------------

def try_openalex(doi: str, session: requests.Session, mailto: str,
                 sleep: float) -> tuple[str, str]:
    """Try OpenAlex best_oa_location PDF + abstract fallback."""
    params = {
        "filter": f"doi:{doi}",
        "select": "id,doi,title,publication_year,best_oa_location,abstract_inverted_index",
        "mailto": mailto,
    }
    try:
        r = session.get(OPENALEX_BASE, params=params, timeout=30)
        time.sleep(sleep)
        if r.status_code != 200:
            return "", ""
        results = r.json().get("results", [])
        if not results:
            return "", ""
        w = results[0]

        # Try PDF from best_oa_location
        boa = w.get("best_oa_location") or {}
        pdf_url = boa.get("pdf_url") or ""
        if pdf_url:
            pdf_bytes = download_pdf(pdf_url, session)
            if pdf_bytes:
                text = pdf_bytes_to_text(pdf_bytes)
                if len(text) > 500:
                    return text, f"openalex_pdf:{pdf_url}"

        # Fallback: abstract
        inv = w.get("abstract_inverted_index") or {}
        abstract = reconstruct_abstract(inv)
        if abstract:
            return abstract, "openalex_abstract"

    except Exception:
        pass
    return "", ""


def try_unpaywall(doi: str, session: requests.Session, email: str,
                  sleep: float) -> tuple[str, str]:
    """Try Unpaywall for OA PDF URL."""
    try:
        r = session.get(
            f"{UNPAYWALL_BASE}/{doi}",
            params={"email": email},
            timeout=20,
        )
        time.sleep(sleep)
        if r.status_code != 200:
            return "", ""
        data = r.json()
        pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf") or ""
        if not pdf_url:
            # Try all locations
            for loc in data.get("oa_locations") or []:
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    break
        if pdf_url:
            pdf_bytes = download_pdf(pdf_url, session)
            if pdf_bytes:
                text = pdf_bytes_to_text(pdf_bytes)
                if len(text) > 500:
                    return text, f"unpaywall_pdf:{pdf_url}"
    except Exception:
        pass
    return "", ""


def try_semantic_scholar(doi: str, session: requests.Session,
                         sleep: float) -> tuple[str, str]:
    """Try Semantic Scholar open access PDF."""
    try:
        r = session.get(
            f"{S2_BASE}/DOI:{doi}",
            params={"fields": "openAccessPdf,abstract"},
            timeout=20,
        )
        time.sleep(sleep)
        if r.status_code != 200:
            return "", ""
        data = r.json()
        pdf_url = (data.get("openAccessPdf") or {}).get("url") or ""
        if pdf_url:
            pdf_bytes = download_pdf(pdf_url, session)
            if pdf_bytes:
                text = pdf_bytes_to_text(pdf_bytes)
                if len(text) > 500:
                    return text, f"s2_pdf:{pdf_url}"
        # Fallback: S2 abstract
        abstract = data.get("abstract") or ""
        if abstract:
            return abstract, "s2_abstract"
    except Exception:
        pass
    return "", ""


# ---------------------------------------------------------------------------
# Main fetch logic for a single paper
# ---------------------------------------------------------------------------

def fetch_paper(doi: str, session: requests.Session, mailto: str,
                sleep: float) -> tuple[str, str]:
    """
    Try all sources in order. Returns (text, source_label).
    source_label is empty string if all failed.
    """
    # 1. OpenAlex (PDF + abstract fallback)
    text, source = try_openalex(doi, session, mailto, sleep)
    if text:
        return text, source

    # 2. Unpaywall
    text, source = try_unpaywall(doi, session, mailto, sleep)
    if text:
        return text, source

    # 3. Semantic Scholar
    text, source = try_semantic_scholar(doi, session, sleep)
    if text:
        return text, source

    return "", "failed"


# ---------------------------------------------------------------------------
# Cluster sampling
# ---------------------------------------------------------------------------

def sample_cluster(df_cluster: pd.DataFrame, n: int,
                   seed: int = 42) -> pd.DataFrame:
    """
    Sample up to n papers from a cluster.
    Stratified by year (5 bins) so coverage isn't skewed to recent years.
    Falls back to simple random sample if stratification fails.
    """
    if len(df_cluster) <= n:
        return df_cluster

    try:
        df_cluster = df_cluster.copy()
        df_cluster["year_bin"] = pd.qcut(
            df_cluster["year"].fillna(2020),
            q=min(5, df_cluster["year"].nunique()),
            duplicates="drop",
            labels=False,
        )
        per_bin = max(1, n // df_cluster["year_bin"].nunique())
        sampled = (
            df_cluster.groupby("year_bin", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed))
        )
        if len(sampled) < n:
            rest = df_cluster[~df_cluster.index.isin(sampled.index)]
            extra = rest.sample(min(len(rest), n - len(sampled)), random_state=seed)
            sampled = pd.concat([sampled, extra])
        return sampled.head(n)
    except Exception:
        return df_cluster.sample(n, random_state=seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch 50 full-text papers per cluster for CE corpus validation."
    )
    ap.add_argument("--corpus", required=True,
                    help="CSV with columns: doi, cluster (and optionally: year, title)")
    ap.add_argument("--out", default="./cluster_fulltexts/",
                    help="Output directory")
    ap.add_argument("--env", default=".env",
                    help="Path to .env file (needs OPENALEX_EMAIL)")
    ap.add_argument("--n", type=int, default=50,
                    help="Papers to fetch per cluster")
    ap.add_argument("--clusters", nargs="*", default=None,
                    help="Cluster IDs to process (e.g. C2 C3 C10). "
                         "Default: all clusters found in corpus.")
    ap.add_argument("--sleep", type=float, default=None,
                    help="Override sleep between requests (seconds)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling")
    args = ap.parse_args()

    # Load .env
    env = load_env(Path(args.env))
    mailto = env.get("OPENALEX_EMAIL")
    if not mailto:
        print(f"ERROR: OPENALEX_EMAIL not set in {args.env}", file=sys.stderr)
        return 1
    sleep_s = args.sleep if args.sleep is not None else float(
        env.get("OPENALEX_SLEEP", "0.3"))
    print(f"Config: mailto={mailto}, sleep={sleep_s}s, n={args.n}")

    # Load corpus
    print(f"\nReading corpus: {args.corpus}")
    df = pd.read_csv(args.corpus)
    df["doi"] = df["doi"].apply(normalise_doi)
    df = df[df["doi"] != ""].drop_duplicates(subset=["doi"])

    if "cluster" not in df.columns:
        print("ERROR: corpus CSV must have a 'cluster' column", file=sys.stderr)
        return 1

    clusters_available = sorted(df["cluster"].dropna().unique().tolist())
    clusters_to_run = args.clusters if args.clusters else clusters_available
    print(f"Clusters found   : {clusters_available}")
    print(f"Clusters to fetch: {clusters_to_run}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "fetch_summary.csv"
    summary_header = ["doi", "cluster", "source", "chars", "is_fulltext"]
    summary_written = set()

    # Load existing summary for resumability
    if summary_path.exists():
        try:
            existing = pd.read_csv(summary_path, usecols=["doi"])
            summary_written = set(existing["doi"].astype(str))
            print(f"Resuming: {len(summary_written)} papers already in summary")
        except Exception:
            pass

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"ce-fulltext-fetch/1.0 (mailto:{mailto})"
    })

    total_ok = 0
    total_fail = 0

    for cluster_id in clusters_to_run:
        df_cl = df[df["cluster"] == cluster_id].copy()
        if df_cl.empty:
            print(f"\n[{cluster_id}] No papers found — skipping")
            continue

        cluster_dir = out / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)

        sample = sample_cluster(df_cl, args.n, seed=args.seed)
        print(f"\n[{cluster_id}] {len(df_cl)} papers in cluster → "
              f"sampling {len(sample)}")

        for i, row in enumerate(sample.itertuples(), 1):
            doi = row.doi
            slug = doi_to_slug(doi)
            txt_path  = cluster_dir / f"{slug}.txt"
            meta_path = cluster_dir / f"{slug}.meta.json"

            # Skip if already done
            if doi in summary_written and txt_path.exists():
                print(f"  [{i}/{len(sample)}] SKIP  {doi}")
                continue

            print(f"  [{i}/{len(sample)}] {doi} ... ", end="", flush=True)
            text, source = fetch_paper(doi, session, mailto, sleep_s)

            chars = len(text)
            is_fulltext = "pdf" in source and chars > 1000

            # Write text file
            txt_path.write_text(text, encoding="utf-8")

            # Write meta JSON
            meta = {
                "doi": doi,
                "cluster": cluster_id,
                "title": getattr(row, "title", ""),
                "year": getattr(row, "year", None),
                "source": source,
                "chars": chars,
                "is_fulltext": is_fulltext,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

            # Append to summary CSV
            new_file = not summary_path.exists()
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(summary_header)
                w.writerow([doi, cluster_id, source, chars, is_fulltext])

            summary_written.add(doi)

            if source == "failed" or not text:
                total_fail += 1
                print(f"FAIL")
            else:
                total_ok += 1
                label = "FULL" if is_fulltext else "ABST"
                print(f"{label}  {chars:,} chars  [{source.split(':')[0]}]")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Done.  Success: {total_ok}  |  Failed: {total_fail}")
    print(f"Summary CSV: {summary_path}")

    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        print("\nPer-cluster breakdown:")
        print(df_sum.groupby("cluster").agg(
            total=("doi", "count"),
            fulltext=("is_fulltext", "sum"),
            abstract_only=("is_fulltext", lambda x: (~x).sum()),
            failed=("source", lambda x: (x == "failed").sum()),
        ).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
