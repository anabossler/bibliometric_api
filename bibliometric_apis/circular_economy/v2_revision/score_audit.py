import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score

h = pd.read_csv("audit_sheet_blind.csv", keep_default_na=False)
k = pd.read_csv("audit_key_llm.csv", keep_default_na=False)
m = h.merge(k, on=["kind","term"])
m["human_category"] = m.human_category.astype(str).str.strip()
m = m[(m.human_category != "") & (m.human_category.str.lower() != "nan")]

if len(m) == 0:
    raise SystemExit("Todavia no hay nada codificado. Corre: python code_audit.py")

for kind, g in m.groupby("kind"):
    tot = len(h[h.kind == kind])
    print(f"\n===== {kind}: {len(g)}/{tot} codificados =====")
    if len(g) < 10:
        print("  (pocos aun, sigue codificando)"); continue
    print(classification_report(g.human_category, g.category, zero_division=0))
    print(f"Cohen kappa: {cohen_kappa_score(g.human_category, g.category):.3f}")
    print("\nconfusion (filas=humano, cols=LLM):")
    print(pd.crosstab(g.human_category, g.category).to_string())
