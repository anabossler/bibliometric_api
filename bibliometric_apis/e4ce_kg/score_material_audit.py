"""Score the material family audit.

Compares each human annotation sheet with the LLM labels in the key and,
when two sheets are given, the annotators with each other. Rows are matched
by item; a changed term text is reported but does not stop the scoring.

Usage:
  python score_material_audit.py audit_material_key.csv A.csv [B.csv]
Writes audit_material_disagreements.csv with every row where labels differ.
"""
import math
import sys

import pandas as pd


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(x, y):
    """Cohen's kappa for two label series of equal length."""
    po = (x.values == y.values).mean()
    px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
    pe = sum(px.get(k, 0) * py.get(k, 0) for k in set(px.index) | set(py.index))
    return (po - pe) / (1 - pe)


def report(name, x, y, w):
    k = int((x.values == y.values).sum())
    n = len(x)
    lo, hi = wilson(k, n)
    agree_w = float((w * (x.values == y.values)).sum() / w.sum())
    print(f"{name}: {k}/{n} = {k / n:.4f} (IC95 {lo:.4f}-{hi:.4f}); "
          f"kappa = {kappa(x, y):.4f}; ponderado por papers = {agree_w:.4f}")


key_path, *ann_paths = sys.argv[1:]
if not ann_paths:
    raise SystemExit(__doc__)
key = pd.read_csv(key_path, keep_default_na=False)
df = key.rename(columns={"category": "llm"})
for i, path in enumerate(ann_paths):
    tag = "ABCDEFG"[i]
    a = pd.read_csv(path, keep_default_na=False)
    if a.family_annotator.eq("").any():
        raise SystemExit(f"{path}: hay filas sin anotar")
    changed = a.merge(key, on="item", suffixes=("", "_key")).query("term != term_key")
    for _, r in changed.iterrows():
        print(f"aviso {path}: item {r['item']} dice '{r['term']}', la clave dice '{r['term_key']}'")
    df = df.merge(a[["item", "family_annotator", "notes"]]
                  .rename(columns={"family_annotator": tag, "notes": f"notes_{tag}"}), on="item")

tags = [c for c in "ABCDEFG" if c in df.columns]
w = df.papers_corpus
print(f"items: {len(df)}")
for t in tags:
    report(f"{t} vs modelo", df[t], df.llm, w)
if len(tags) >= 2:
    report("A vs B", df.A, df.B, w)

cols = ["item", "term", "papers_corpus", "llm"] + tags + [f"notes_{t}" for t in tags]
mask = pd.concat([df[t] != df.llm for t in tags], axis=1).any(axis=1)
df.loc[mask, cols].to_csv("audit_material_disagreements.csv", index=False)
print(f"desacuerdos guardados: {int(mask.sum())} filas en audit_material_disagreements.csv")
