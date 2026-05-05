"""

Test whether the practice-to-talk pattern reported in the paper (computed on
abstracts) holds when computed on FULL TEXT, at the mega-area level.

Usage:
  python validate_fulltext_ratio.py \\
      --summary  ./cluster_fulltexts/fetch_summary.csv \\
      --fulltext ./cluster_fulltexts/ \\
      --corpus   ./corpus_with_clusters.csv \\
      --out      ./results/ce_review/fulltext_validation.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Vocabularies (identical to paper Appendix A)
# ---------------------------------------------------------------------------

TALK_RE = re.compile(
    r"circular econom[a-z]*"
    r"|circularity"
    r"|closed.?loop"
    r"|sustainab[a-z]+ paradigm"
    r"|sustainability transition"
    r"|CE framework"
    r"|CE strateg[a-z]+"
    r"|circular business model",
    re.IGNORECASE,
)

PRACTICE_RE = re.compile(
    r"recycl[a-z]+"
    r"|remanufactur[a-z]+"
    r"|refurbish[a-z]+"
    r"|repair[a-z]*"
    r"|reus[a-z]+"
    r"|reduc[a-z]+ waste"
    r"|waste reduction"
    r"|upcycl[a-z]+"
    r"|life.?cycle assessment"
    r"|\bLCA\b"
    r"|end.?of.?life"
    r"|material recovery"
    r"|energy recovery"
    r"|reverse logistics"
    r"|mechanical recycling"
    r"|chemical recycling"
    r"|biodegrad[a-z]+"
    r"|compost[a-z]+",
    re.IGNORECASE,
)


def ratio(text):
    """Returns (practice_to_talk_ratio, talk_count, practice_count)."""
    if not text:
        return 0.0, 0, 0
    talk = len(TALK_RE.findall(text))
    prac = len(PRACTICE_RE.findall(text))
    return prac / max(talk, 1), talk, prac


# ---------------------------------------------------------------------------
# Cluster -> mega-area (paper Table 1)
# ---------------------------------------------------------------------------

CLUSTER_TO_MEGA = {
    "C10": "Materials & Technical Recycling",
    "C11": "Materials & Technical Recycling",
    "C2":  "Energy & Resource Systems",
    "C14": "Energy & Resource Systems",
    "C7":  "Industrial Sectors & Applied Cases",
    "C8":  "Industrial Sectors & Applied Cases",
    "C9":  "Industrial Sectors & Applied Cases",
    "C4":  "Business, Policy & Governance",
    "C6":  "Business, Policy & Governance",
    "C3":  "Sustainability Framing & Society",
    "C5":  "Sustainability Framing & Society",
}

PAPER_RATIOS = {
    "Materials & Technical Recycling":    2.07,
    "Energy & Resource Systems":          1.34,
    "Industrial Sectors & Applied Cases": 1.21,
    "Sustainability Framing & Society":   0.52,
    "Business, Policy & Governance":      0.22,
}


def classify(r):
    if pd.isna(r):
        return "n/a"
    if r >= 1.5:
        return "Practice-oriented"
    if r >= 0.8:
        return "Balanced"
    if r >= 0.3:
        return "Talk-leaning"
    return "Talk-dominant"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary",  required=True)
    ap.add_argument("--fulltext", required=True)
    ap.add_argument("--corpus",   required=True)
    ap.add_argument("--out",      default="./fulltext_validation.csv")
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    ft_only = summary[
        summary["is_fulltext"].astype(str).str.lower() == "true"
    ].copy()
    print("Full-text papers available: " + str(len(ft_only)))
    print("  Per cluster: " + str(dict(ft_only["cluster"].value_counts().sort_index())))

    corpus = pd.read_csv(args.corpus)
    corpus["doi"] = corpus["doi"].astype(str).str.lower().str.strip()
    corpus = corpus[["doi", "abstract"]].dropna(subset=["abstract"])
    abst_lookup = dict(zip(corpus["doi"], corpus["abstract"]))

    root = Path(args.fulltext)

    # -----------------------------------------------------------------------
    # Compute paper-level ratios
    # -----------------------------------------------------------------------
    rows = []
    missing_files = 0
    no_abstract = 0

    for _, row in ft_only.iterrows():
        doi     = str(row["doi"]).lower().strip()
        cluster = str(row["cluster"])
        slug    = re.sub(r"[^\w\-]", "_", doi)[:120]
        txt_path = root / ("cluster_" + cluster) / (slug + ".txt")

        if not txt_path.exists():
            missing_files += 1
            continue

        full_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if len(full_text) < 1000:
            continue

        abstract = abst_lookup.get(doi, "")
        if not abstract:
            no_abstract += 1
            continue

        r_full, t_full, p_full = ratio(full_text)
        r_abs,  t_abs,  p_abs  = ratio(abstract)

        rows.append({
            "doi":          doi,
            "cluster":      cluster,
            "mega_area":    CLUSTER_TO_MEGA.get(cluster, "Unknown"),
            "chars_full":   len(full_text),
            "chars_abst":   len(abstract),
            "talk_full":    t_full,
            "prac_full":    p_full,
            "ratio_full":   r_full,
            "talk_abst":    t_abs,
            "prac_abst":    p_abs,
            "ratio_abst":   r_abs,
            "delta":        r_full - r_abs,
        })

    if missing_files:
        print("WARNING: " + str(missing_files) + " .txt files not found on disk")
    if no_abstract:
        print("WARNING: " + str(no_abstract) + " papers had no matching abstract in corpus")

    df = pd.DataFrame(rows)
    print("Papers analysed: " + str(len(df)) + "\n")

    if df.empty:
        print("ERROR: no data to analyse", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # PAPER-LEVEL
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PAPER-LEVEL: do abstract and full-text ratios agree?")
    print("=" * 70)

    pearson_r,  pearson_p  = stats.pearsonr(df["ratio_abst"], df["ratio_full"])
    spearman_r, spearman_p = stats.spearmanr(df["ratio_abst"], df["ratio_full"])

    print("  Pearson  (abstract vs full):  r   = {:+.3f}  (p = {:.2e})".format(
        pearson_r, pearson_p))
    print("  Spearman (abstract vs full):  rho = {:+.3f}  (p = {:.2e})".format(
        spearman_r, spearman_p))

    print("\n  Delta (ratio_full - ratio_abst) by mega-area:")
    delta_stats = df.groupby("mega_area")["delta"].agg(
        ["mean", "median", "std", "count"]).round(3)
    print(delta_stats.to_string())

    # -----------------------------------------------------------------------
    # MEGA-AREA LEVEL
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MEGA-AREA LEVEL: does the paper's ranking hold under full text?")
    print("=" * 70)

    mega = df.groupby("mega_area").agg(
        n            =("doi",        "count"),
        ratio_abst   =("ratio_abst", "mean"),
        ratio_full   =("ratio_full", "mean"),
        ratio_full_se=("ratio_full",
                       lambda x: x.std(ddof=1) / max(len(x) ** 0.5, 1)),
    ).round(3)
    mega["paper_ratio_table3"] = mega.index.map(PAPER_RATIOS)
    mega = mega.sort_values("ratio_full", ascending=False)

    mega["profile_full"]  = mega["ratio_full"].apply(classify)
    mega["profile_abst"]  = mega["ratio_abst"].apply(classify)
    mega["profile_paper"] = mega["paper_ratio_table3"].apply(classify)
    mega["profile_match_paper"] = (
        mega["profile_full"] == mega["profile_paper"])

    print(mega.to_string())

    # Rank tests
    rank_full  = mega["ratio_full"].rank(ascending=False)
    rank_abst  = mega["ratio_abst"].rank(ascending=False)
    rank_paper = mega["paper_ratio_table3"].rank(ascending=False)

    rho_full_abst,  p_fa = stats.spearmanr(rank_full, rank_abst)
    rho_full_paper, p_fp = stats.spearmanr(rank_full, rank_paper)

    print("\n  Spearman rank correlation:")
    print("    full-text  vs  this-sample-abstract     : "
          "rho = {:+.3f}  (p = {:.3f})".format(rho_full_abst, p_fa))
    print("    full-text  vs  paper Table 3 (full corpus): "
          "rho = {:+.3f}  (p = {:.3f})".format(rho_full_paper, p_fp))

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if rho_full_paper >= 0.9:
        verdict = ("ROBUST. The mega-area ranking under full-text computation "
                   "matches the paper's abstract-level ranking. The "
                   "practice-to-talk asymmetry is NOT an artefact of "
                   "abstract-writing conventions.")
    elif rho_full_paper >= 0.7:
        verdict = ("MOSTLY ROBUST. The qualitative ordering is preserved with "
                   "minor reordering of intermediate positions. The headline "
                   "finding (Materials > ... > Business) holds; intermediate "
                   "claims should be hedged.")
    elif rho_full_paper >= 0.4:
        verdict = ("PARTIALLY ROBUST. The ranking changes meaningfully under "
                   "full-text computation. The paper's claims should be "
                   "RE-FRAMED as abstract-level findings, with a stronger "
                   "limitations section.")
    else:
        verdict = ("NOT ROBUST. The full-text ranking diverges substantially "
                   "from the abstract-level ranking. The headline finding is "
                   "in serious trouble: the paper documents an artefact of "
                   "abstract-writing conventions, not a structural property "
                   "of the literature.")

    print(verdict)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\nPaper-level results: " + str(out_path))

    mega_path = out_path.parent / "fulltext_validation_by_megaarea.csv"
    mega.to_csv(mega_path)
    print("Mega-area summary  : " + str(mega_path))

    # -----------------------------------------------------------------------
    # Appendix F draft (printed without f-strings to avoid backslash issues)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("APPENDIX F DRAFT")
    print("=" * 70)

    counts_str = ", ".join(
        m.split(" & ")[0] + "=" + str(n)
        for m, n in df["mega_area"].value_counts().items()
    )

    intro = (
        "\\section{Full-text validation of the practice-to-talk ratio}\n"
        "\\label{app:fulltext}\n\n"
        "Because abstracts may not faithfully reflect the operational content "
        "of the full paper, we tested whether the mega-area ranking by "
        "practice-to-talk ratio holds when the ratio is computed on full body "
        "text rather than on abstracts.\n\n"
        "We retrieved Open Access PDFs for a stratified sample of papers per "
        "cluster, extracted full text, and computed the practice-to-talk "
        "ratio (Appendix A vocabularies) on (i) the corpus abstract and "
        "(ii) the full document body for each paper. Of the targeted papers, "
        + str(len(df)) + " had retrievable full text and a matching abstract "
        "in the corpus, distributed as: " + counts_str + ".\n\n"
        "At the paper level, the abstract-level and full-text ratios are "
        "positively correlated (Pearson r = " + "{:.2f}".format(pearson_r) +
        ", p = " + "{:.2e}".format(pearson_p) + "; Spearman rho = " +
        "{:.2f}".format(spearman_r) + ", p = " + "{:.2e}".format(spearman_p) +
        "), confirming that the two estimates track the same underlying "
        "signal.\n\n"
        "At the mega-area level, the ranking by mean practice-to-talk ratio "
        "under full-text computation is compared against the ranking reported "
        "in the main text (Table~\\ref{tab:talkvspractice}). The Spearman "
        "rank correlation between the two rankings is rho = " +
        "{:.2f}".format(rho_full_paper) + " (p = " +
        "{:.3f}".format(p_fp) + ")."
    )
    print(intro)
    print()

    # Tabular block — built as plain string concatenation (no f-strings)
    BS = chr(92)        # backslash
    DBS = BS + BS       # \\
    AMP = " & "

    print(BS + "begin{table}[ht]")
    print(BS + "centering")
    print(BS + "caption{Mega-area practice-to-talk ratios under abstract vs "
          "full-text computation (n = " + str(len(df)) +
          " papers with retrievable Open Access full text).}")
    print(BS + "label{tab:rep_fulltext}")
    print(BS + "small")
    print(BS + "begin{tabular}{lrrrrl}")
    print(BS + "toprule")
    print(BS + "textbf{Mega-area}" + AMP +
          BS + "textbf{n}" + AMP +
          BS + "textbf{Abstract}" + AMP +
          BS + "textbf{Full text}" + AMP +
          BS + "textbf{Profile (full)}" + AMP +
          BS + "textbf{Match} " + DBS)
    print(BS + "midrule")

    for mega_name, r in mega.iterrows():
        match = "Yes" if r["profile_match_paper"] else "No"
        short = mega_name.split(" & ")[0]
        line = (
            short + AMP +
            str(int(r["n"])) + AMP +
            "{:.2f}".format(r["ratio_abst"]) + AMP +
            "{:.2f}".format(r["ratio_full"]) + AMP +
            r["profile_full"] + AMP +
            match + " " + DBS
        )
        print(line)

    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")

    print()
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
