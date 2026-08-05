import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

h   = pd.read_csv("audit_sheet_blind.csv", keep_default_na=False)
k   = pd.read_csv("audit_key_llm.csv", keep_default_na=False)
art = pd.read_csv("tmo_article_level.csv")

rule = (art[art.obj_class.isin(['operational','discursive','unclassified'])]
        .groupby('obj_name').obj_class.agg(lambda s: s.mode().iat[0])
        .rename('rule_class').reset_index().rename(columns={'obj_name':'term'}))

o = h.merge(k, on=['kind','term']).merge(rule, on='term', how='left')
o = o[o.kind == 'objective'].copy()
print("objetivos auditados con rule_class:", o.rule_class.notna().sum(), "/", len(o))

o['human_bin'] = (o.human_category == 'material_technical').astype(int)
o['llm_bin']   = (o.category       == 'material_technical').astype(int)
o['rule_bin']  = (o.rule_class     == 'operational').astype(int)

print("kappa humano vs LLM  (8 clases):", round(cohen_kappa_score(o.human_category, o.category), 3))
print("kappa humano vs LLM  (binaria) :", round(cohen_kappa_score(o.human_bin, o.llm_bin), 3))
print("kappa humano vs REGLA (binaria):", round(cohen_kappa_score(o.human_bin, o.rule_bin), 3))
print(pd.crosstab(o.human_bin, o.rule_bin))
print(classification_report(o.human_bin, o.rule_bin, zero_division=0))
