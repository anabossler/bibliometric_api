"""Build a blind audit sample for the material family classification.

Draws a stratified random sample of Material terms used by corpus papers
(corpus_relevante = true), excluding terms shown as examples in the prompt.
Writes a blind sheet for annotators and a separate key with the LLM labels.
Usage: export NEO4J_PW=...; python make_material_audit.py
"""
import os
import re

import pandas as pd
from neo4j import GraphDatabase

SEED, PER_FAMILY = 42, 6
Q = """MATCH (p:Paper)-[:TARGETS_MATERIAL]->(m:Material)
WHERE p.corpus_relevante = true
RETURN m.name AS term, count(p) AS papers_corpus"""

drv = GraphDatabase.driver("bolt://localhost:7691", auth=("neo4j", os.environ["NEO4J_PW"]))
with drv.session(database="circulareconomy") as s:
    corpus = pd.DataFrame([dict(r) for r in s.run(Q)])
drv.close()

examples = set(re.findall(r'^"(.+?)"\s+->', open("prompt_classify_material.txt").read(), re.M))
d = pd.read_csv("tmo_material_classified.csv", keep_default_na=False)
d = d.merge(corpus, on="term", how="inner")
d = d[~d.term.isin(examples)]

sample = (d.sample(frac=1, random_state=SEED).groupby("category").head(PER_FAMILY)
            .sample(frac=1, random_state=SEED).reset_index(drop=True))
sample.insert(0, "item", range(1, len(sample) + 1))

sample[["item", "term", "papers_corpus"]].assign(family_annotator="", notes="") \
    .to_csv("audit_material_blind.csv", index=False)
sample[["item", "term", "papers_corpus", "category"]] \
    .to_csv("audit_material_key.csv", index=False)
print(f"{len(sample)} terminos, {sample.category.nunique()} familias, "
      f"{len(examples)} ejemplos del prompt excluidos")
