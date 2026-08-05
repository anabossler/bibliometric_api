"""Analisis geografico con observaciones INDEPENDIENTES (corpus completo).

El chi2 publicado usa filas paper x pais, que no son independientes. Aqui se
recalcula de tres formas sobre los 20.066 papers:

  A) nivel articulo, 3 categorias excluyentes (Solo Norte / Solo Sur / Colab N-S)
  B) nivel articulo, binaria (algun autor del Sur si/no)
  C) conteo fraccionado (1/k por pais) vs conteo entero

Join directo: ce_papers_meta -> ce_affiliations -> ce_institutions
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BASE = Path("ce_openalex_v2")

NORTH = set("""AL AD AT BY BE BA BG HR CY CZ DK EE FO FI FR DE GI GR GG HU IS IE IM
IT JE LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SJ SE CH UA GB
VA AX US CA BM GL PM AU NZ JP KR IL""".split())

nid = lambda x: (pd.Series(x).astype(str).str.strip()
                 .str.replace(r"^https?://openalex\.org/", "", regex=True).str.upper())
ndoi = lambda x: (pd.Series(x).astype(str).str.strip().str.lower()
                  .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True))

art = pd.read_csv("tmo_article_level.csv", usecols=["doi", "mega"]).drop_duplicates()
art["doi_n"] = ndoi(art.doi)
print(f"corpus analitico: {len(art):,} papers, {art.mega.nunique()} mega-areas\n")

papers = pd.read_csv(BASE / "ce_papers_meta.csv", usecols=["openalex_id", "doi"])
aff    = pd.read_csv(BASE / "ce_affiliations.csv", usecols=["paper_id", "institution_id"])
inst   = pd.read_csv(BASE / "ce_institutions.csv", usecols=["openalex_id", "country"])
papers["doi_n"] = ndoi(papers.doi); papers["pid"] = nid(papers.openalex_id)
aff["pid"] = nid(aff.paper_id);     aff["iid"] = nid(aff.institution_id)
inst["iid"] = nid(inst.openalex_id)

pc = (papers[papers.doi_n.isin(set(art.doi_n))][["doi_n", "pid"]]
        .merge(aff[["pid", "iid"]].drop_duplicates(), on="pid")
        .merge(inst[["iid", "country"]].dropna(), on="iid")
        [["doi_n", "country"]].drop_duplicates())
pc["cc"] = pc.country.astype(str).str.strip().str.upper()
pc["north"] = pc.cc.isin(NORTH)

m = pc.merge(art[["doi_n", "mega"]], on="doi_n")
print(f"pares unicos paper x pais : {len(m):,}   (publicado: 23.606)")
print(f"papers con pais           : {m.doi_n.nunique():,}   (publicado: 16.812)")
print(f"paises distintos          : {m.cc.nunique()}")
print("Top-12 SUR                : " + ", ".join(f"{k}({v})" for k, v in m[~m.north].cc.value_counts().head(12).items()))
if m.mega.nunique() < 5:
    sys.exit(f"\nABORTADO: solo {m.mega.nunique()} mega-area(s).")

print("\n" + "=" * 70 + "\nVALIDACION: reproducir la tabla paper x pais publicada\n" + "=" * 70)
pub = {"Industrial Sectors & Applied Cases": (3675, 1515),
       "Business, Policy & Governance": (4460, 1974),
       "Energy & Resource Systems": (2840, 1321),
       "Materials & Technical Recycling": (1947, 1015),
       "Sustainability Framing & Society": (3017, 1842)}
got = m.groupby(["mega", "north"]).size().unstack(fill_value=0)
ok = True
for k, (pn, ps) in pub.items():
    if k not in got.index:
        print(f"  {k[:36]:36s} AUSENTE"); ok = False; continue
    gn, gs = int(got.loc[k, True]), int(got.loc[k, False])
    d = "OK" if abs(gn-pn) <= max(25, .03*pn) and abs(gs-ps) <= max(25, .03*ps) else "DIFIERE"
    ok &= (d == "OK")
    print(f"  {k[:36]:36s} N {gn:5d} (pub {pn:5d})   S {gs:5d} (pub {ps:5d})   {d}")
print("\n  => reproduce la tabla publicada" if ok else "\n  => OJO: revisa el mapeo Norte/Sur")

def report(tab, title):
    chi2, p, dof, _ = chi2_contingency(tab, correction=False)
    N = int(tab.values.sum())
    V = np.sqrt(chi2 / (N * max(1, min(tab.shape[0]-1, tab.shape[1]-1))))
    print(f"\n{title}"); print(tab.to_string())
    print(f"  N = {N:,}   chi2({dof}) = {chi2:.2f}   p = {p:.3e}   Cramer V = {V:.4f}")
    return chi2, p, dof, V, N

print("\n" + "=" * 70 + "\nA) NIVEL ARTICULO, 3 CATEGORIAS EXCLUYENTES\n" + "=" * 70)
per = m.groupby("doi_n").north.agg(["sum", "size"])
per["cat"] = np.where(per["sum"] == per["size"], "North only",
             np.where(per["sum"] == 0, "South only", "North-South collab"))
per = per.join(art.drop_duplicates("doi_n").set_index("doi_n")["mega"])
tabA = pd.crosstab(per.mega, per.cat)
tabA = tabA[[c for c in ["North only", "North-South collab", "South only"] if c in tabA.columns]]
tabA = tabA.loc[(tabA["North only"]/tabA.sum(axis=1)).sort_values(ascending=False).index]
rA = report(tabA, "Recuento de articulos (cada paper cuenta UNA vez):")
print("\n  % por fila:"); print((tabA.div(tabA.sum(axis=1), axis=0)*100).round(1).to_string())

print("\n" + "=" * 70 + "\nB) NIVEL ARTICULO, BINARIA: algun autor del Sur\n" + "=" * 70)
per["any_south"] = per["sum"] < per["size"]
tabB = pd.crosstab(per.mega, per.any_south); tabB.columns = ["No South author", "Has South author"]
tabB = tabB.loc[(tabB.iloc[:,0]/tabB.sum(axis=1)).sort_values(ascending=False).index]
rB = report(tabB, "")
print("\n  % con algun autor del Sur:"); print((tabB.iloc[:,1]/tabB.sum(axis=1)*100).round(1).to_string())

