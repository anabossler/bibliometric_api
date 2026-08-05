import pandas as pd, numpy as np
from scipy.stats import spearmanr

art = pd.read_csv("tmo_article_level.csv")
obj = pd.read_csv("tmo_objective_classified.csv").rename(columns={"term": "obj_name"})
m = art.merge(obj[["obj_name", "category"]], on="obj_name", how="left")
m["mt"] = (m.category == "material_technical")

def tabla(key):
    t = m.groupby(key).agg(n=("doi", "size"), pct_mt=("mt", "mean"))
    s = m.groupby(key)[["practice_count", "talk_count"]].sum()
    t["pct_mt"] = (t.pct_mt * 100).round(1)
    t["lex_ratio"] = (s.practice_count / s.talk_count).round(3)
    return t.sort_values("lex_ratio", ascending=False)

for key, label in [("mega", "5 MEGA-AREAS"), ("cl", "11 CLUSTERS")]:
    t = tabla(key)
    r = spearmanr(t.lex_ratio, t.pct_mt)
    print(f"\n== {label} ==")
    print(t.to_string())
    print(f"Spearman rho={r.statistic:.3f}  p={r.pvalue:.4f}")

# permutacion a nivel paper sobre los 11 clusters
t = tabla("cl"); obs = spearmanr(t.lex_ratio, t.pct_mt).statistic
rng = np.random.default_rng(20260805); null = []
for _ in range(1000):
    sh = m.assign(mt=rng.permutation(m.mt.values)).groupby("cl").mt.mean()
    null.append(spearmanr(t.lex_ratio, sh.reindex(t.index)).statistic)
null = np.array(null)
p1 = (np.sum(np.abs(null) >= abs(obs)) + 1) / 1001
print(f"\n== PERMUTACION (11 clusters, 1000 draws) ==")
print(f"obs rho={obs:.3f} | null 95% = [{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}] | p_plus1={p1:.4f}")
