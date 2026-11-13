"""Score the material family audit.

Compares each human annotation sheet with the LLM labels in the key and,
when two sheets are given, the annotators with each other. Rows are matched
by item; a changed term text is reported but does not stop the scoring.

The sample was drawn with a fixed number of terms per LLM family, so raw
agreement over-represents small families. With --counts and --classified the
script also reports post-stratified estimates for the whole corpus, weighting
each family by its number of corpus terms or corpus papers, with stratified
bootstrap 95% intervals. It also reports agreement on the binary decision
material versus non-material and the estimated share of non-material
annotations according to each annotator.

Usage:
  python score_material_audit.py audit_material_key.csv A.csv [B.csv]
      [--counts tmo_material_corpus_counts.csv --classified tmo_material_classified.csv]
Writes audit_material_disagreements.csv with every row where labels differ.
"""
import argparse
import math

import numpy as np
import pandas as pd

NON_MATERIAL = {"sector_or_concept", "generic_waste_resource"}
SEED, N_BOOT = 42, 2000


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(x, y):
    """Cohen's kappa for two label series of equal length."""
    po = (np.asarray(x) == np.asarray(y)).mean()
    px, py = pd.Series(x).value_counts(normalize=True), pd.Series(y).value_counts(normalize=True)
    pe = sum(px.get(k, 0) * py.get(k, 0) for k in set(px.index) | set(py.index))
    return (po - pe) / (1 - pe)


def report(name, x, y, w):
    hit = np.asarray(x) == np.asarray(y)
    k, n = int(hit.sum()), len(hit)
    lo, hi = wilson(k, n)
    print(f"{name}: {k}/{n} = {k / n:.4f} (IC95 {lo:.4f}-{hi:.4f}); "
          f"kappa = {kappa(x, y):.4f}; ponderado por papers en la muestra = {(w * hit).sum() / w.sum():.4f}")


def poststratified(df, value, weights, by_papers):
    """Estimate a corpus-level mean of `value` from a sample stratified by df.llm.

    Returns the point estimate and a stratified bootstrap 95% interval.
    """
    strata = [(weights[fam], g[value].to_numpy(dtype=float), g.papers_corpus.to_numpy(dtype=float))
              for fam, g in df.groupby("llm")]
    total_w = sum(wt for wt, _, _ in strata)

    def mean(v, p):
        return (v * p).sum() / p.sum() if by_papers else v.mean()

    point = sum(wt * mean(v, p) for wt, v, p in strata) / total_w
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        acc = 0.0
        for wt, v, p in strata:
            idx = rng.integers(0, len(v), len(v))
            acc += wt * mean(v[idx], p[idx])
        boots[b] = acc / total_w
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


ap = argparse.ArgumentParser()
ap.add_argument("key")
ap.add_argument("sheets", nargs="+")
ap.add_argument("--counts")
ap.add_argument("--classified")
args = ap.parse_args()

key = pd.read_csv(args.key, keep_default_na=False)
df = key.rename(columns={"category": "llm"})
tags = []
for i, path in enumerate(args.sheets):
    tag = "ABCDEFG"[i]
    tags.append(tag)
    a = pd.read_csv(path, keep_default_na=False)
    if a.family_annotator.eq("").any():
        raise SystemExit(f"{path}: hay filas sin anotar")
    changed = a.merge(key, on="item", suffixes=("", "_key")).query("term != term_key")
    for _, r in changed.iterrows():
        print(f"aviso {path}: item {r['item']} dice '{r['term']}', la clave dice '{r['term_key']}'")
    df = df.merge(a[["item", "family_annotator", "notes"]]
                  .rename(columns={"family_annotator": tag, "notes": f"notes_{tag}"}), on="item")

w = df.papers_corpus
print(f"items: {len(df)}")
for t in tags:
    report(f"{t} vs modelo", df[t], df.llm, w)
if len(tags) >= 2:
    report("A vs B", df.A, df.B, w)

df["llm_nonmat"] = df.llm.isin(NON_MATERIAL)
for t in tags:
    df[f"{t}_nonmat"] = df[t].isin(NON_MATERIAL)
    report(f"{t} vs modelo (material / no material)", df[f"{t}_nonmat"], df.llm_nonmat, w)

if args.counts and args.classified:
    counts = pd.read_csv(args.counts, keep_default_na=False)
    cls = pd.read_csv(args.classified, keep_default_na=False)[["term", "category"]]
    corpus = cls.merge(counts, on="term", how="inner")
    corpus = corpus[corpus.papers_corpus > 0]
    w_terms = corpus.groupby("category").size().to_dict()
    w_papers = corpus.groupby("category").papers_corpus.sum().to_dict()
    nonmat_llm = sum(v for k, v in w_papers.items() if k in NON_MATERIAL) / sum(w_papers.values())
    print(f"corpus: {len(corpus)} terminos, {sum(w_papers.values())} anotaciones de material; "
          f"no material segun el modelo = {nonmat_llm:.4f}")
    for t in tags:
        df[f"{t}_hit"] = df[t] == df.llm
        for label, wts, byp in (("por termino", w_terms, False), ("por papers", w_papers, True)):
            p, lo, hi = poststratified(df, f"{t}_hit", wts, byp)
            print(f"{t} vs modelo, postestratificado {label}: {p:.4f} (IC95 bootstrap {lo:.4f}-{hi:.4f})")
        p, lo, hi = poststratified(df, f"{t}_nonmat", w_papers, True)
        print(f"{t}: no material estimado (por papers): {p:.4f} (IC95 bootstrap {lo:.4f}-{hi:.4f})")

cols = ["item", "term", "papers_corpus", "llm"] + tags + [f"notes_{t}" for t in tags]
mask = pd.concat([df[t] != df.llm for t in tags], axis=1).any(axis=1)
df.loc[mask, cols].to_csv("audit_material_disagreements.csv", index=False)
print(f"desacuerdos guardados: {int(mask.sum())} filas en audit_material_disagreements.csv")
