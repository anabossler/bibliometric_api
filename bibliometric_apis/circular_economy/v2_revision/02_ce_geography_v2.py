"""
WIDENED VERSION  OF 02_ce_geography_v2.py

CHANGE: use ce_affiliations.csv with paper_id (output of the FIXED fetch).

Run:
  python 02_ce_geography_v3.py \
      --topics ./results_circular_economy/full_corpus/paper_topics_clean.csv \
      --base ./ce_openalex_v2/ \
      --out results/ce_review_v3/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


MEGA_AREAS = {
    "Materials & Technical Recycling": [10, 11],
    "Industrial Sectors & Applied Cases": [7, 8, 9],
    "Energy & Resource Systems": [2, 14],
    "Business, Policy & Governance": [4, 6],
    "Sustainability Framing & Society": [3, 5],
}
CLUSTER_TO_MEGA = {c: m for m, cs in MEGA_AREAS.items() for c in cs}

GLOBAL_NORTH = {
    "AD","AL","AT","BA","BE","BG","BY","CH","CY","CZ","DE","DK","EE","ES","FI",
    "FO","FR","GB","GG","GI","GR","HR","HU","IE","IM","IS","IT","JE","LI","LT",
    "LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE",
    "SI","SK","SM","UA","VA","XK","CA","US","AU","NZ","JP","KR","IL",
}


def classify(country):
    if country is None or pd.isna(country):
        return "Unknown"
    return "Global North" if str(country).upper() in GLOBAL_NORTH else "Global South"


def normalise_doi(s):
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True))


def load_topics(path):
    df = pd.read_csv(path, on_bad_lines="skip", dtype={"cluster": "Int64"})
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)
    df["mega_area"] = df["cluster"].map(CLUSTER_TO_MEGA)
    df = df.dropna(subset=["mega_area"])
    return df[["doi", "cluster", "mega_area"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--base", default="ce_openalex_v2/")
    ap.add_argument("--out", default="results/ce_review_v3/")
    ap.add_argument("--top-countries-n", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)

    print("Loading...")
    topics = load_topics(Path(args.topics))
    papers = pd.read_csv(base / "ce_papers_meta.csv")
    aff = pd.read_csv(base / "ce_affiliations.csv")
    inst = pd.read_csv(base / "ce_institutions.csv")

    # --- comprobacion: aff DEBE tener paper_id ---
    if "paper_id" not in aff.columns:
        sys.exit("ERROR: ce_affiliations.csv no tiene columna paper_id.\n"
                 "Estas usando el fichero antiguo. Re-corre el fetch FIXED.")

    print(f"  topics: {len(topics):,} | papers: {len(papers):,} | "
          f"affiliations: {len(aff):,} | institutions: {len(inst):,}")

    # Step 1: doi -> openalex_id (paper_id)
    print("\n[1/3] topics -> papers (doi -> paper_id)...")
    topics["doi_norm"] = normalise_doi(topics["doi"])
    papers["doi_norm"] = normalise_doi(papers["doi"])
    psub = papers[["openalex_id", "doi_norm"]].drop_duplicates("doi_norm")
    t = topics.merge(psub, on="doi_norm", how="inner").rename(
        columns={"openalex_id": "paper_id"})
    print(f"  matched: {len(t):,} ({100*len(t)/len(topics):.1f}% of topics)")

    # Step 2: paper_id + institution_id (DIRECTO, ya no propaga por autor)
    print("\n[2/3] join afiliaciones a nivel paper (correcto)...")
    pi = t[["paper_id", "mega_area"]].merge(
        aff[["paper_id", "institution_id"]].drop_duplicates(),
        on="paper_id", how="inner")
    print(f"  paper-institution rows: {len(pi):,}")

    # Step 3: institution -> country
    print("\n[3/3] join pais...")
    inst_c = inst[["openalex_id", "country"]].rename(
        columns={"openalex_id": "institution_id"})
    pic = pi.merge(inst_c, on="institution_id", how="inner")

    paper_country = (pic[["paper_id", "mega_area", "country"]]
                     .drop_duplicates().dropna(subset=["country"]))
    paper_country = paper_country[
        paper_country["country"].astype(str).str.len() > 0]
    paper_country["region"] = paper_country["country"].apply(classify)

    n_with = paper_country["paper_id"].nunique()
    n_tot = t["paper_id"].nunique()
    print(f"  papers con pais: {n_with:,} ({100*n_with/n_tot:.1f}%)")

    # --- Tabla por pais x mega-area
    counts = (paper_country.groupby(["mega_area", "country"]).size()
              .reset_index(name="n_papers")
              .sort_values(["mega_area", "n_papers"], ascending=[True, False]))
    top = counts.groupby("mega_area").head(args.top_countries_n)
    top.to_csv(out / "geography_by_area_v3.csv", index=False)

    # --- North/South
    rc = (paper_country.groupby(["mega_area", "region"]).size()
          .reset_index(name="n"))
    piv = rc.pivot(index="mega_area", columns="region", values="n").fillna(0)
    for col in ("Global North", "Global South", "Unknown"):
        if col not in piv.columns: piv[col] = 0
    piv["total"] = piv["Global North"] + piv["Global South"]
    piv["pct_north"] = (100*piv["Global North"]/piv["total"]).round(1)
    piv["pct_south"] = (100*piv["Global South"]/piv["total"]).round(1)
    piv = piv[["Global North","Global South","total","pct_north","pct_south"]]
    piv = piv.sort_values("pct_north", ascending=False)
    piv.to_csv(out / "geography_global_split_v3.csv")

    print("\n=== Tabla 4 RECALCULADA (atribucion a nivel paper) ===")
    print(piv.to_string())

    # --- VERIFICACION INDONESIA
    sf = "Sustainability Framing & Society"
    idn = paper_country[(paper_country.mega_area==sf) &
                        (paper_country.country.str.upper()=="ID")]["paper_id"].nunique()
    print(f"\n>>> Indonesia en Sustainability Framing (v3 correcta): {idn} "
          f"(antes/v2: 427) <<<")
    topc = (paper_country[paper_country.mega_area==sf].groupby("country")["paper_id"]
            .nunique().sort_values(ascending=False).head(6))
    print("Top 6 paises:"); print(topc.to_string())

    print(f"\nDONE -> {out}")
    print("Compara geography_global_split_v3.csv con la Tabla 4 del paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
