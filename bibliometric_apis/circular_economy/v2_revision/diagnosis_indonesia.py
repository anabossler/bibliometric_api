
"""DIAGNOSTICO DEL "INDONESIA PARADOX"
Examina si la concentracion de Indonesia en Sustainability Framing es un hallazgo
real o un artefacto de cobertura/atribucion de OpenAlex.

Addresses four hypotheses raised by Reviewer 1 regarding publication cultures, database coverage, language effects, and attribution conventions:

H1 (Concentration):
Is Indonesia disproportionately represented in Sustainability
Framing, or does it simply have high publication output across
the corpus as a whole?

H2 (Textual Profile):
Are Indonesian abstracts within this cluster shorter or more
generic than the corpus average?

H3 (Attribution Patterns):
Are Indonesian papers more frequently associated with
single-country authorship (resulting in unambiguous attribution
to Indonesia) compared with internationally collaborative papers
from Western countries?

H4 (Coverage and Publication Type):
What proportion of Indonesian output consists of conference
proceedings or locally oriented publications versus journal
articles, and how do open-access rates and citation patterns
compare across countries?


Needed files:
    paper_topics_clean.csv   ce_papers_meta.csv   ce_authorships.csv
    ce_affiliations.csv      ce_institutions.csv
    corpus_with_clusters.csv  (para H2: tiene 'abstract')

Execute:  python diagnostico_indonesia.py

"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

MEGA = {10:"Materials & Technical Recycling",11:"Materials & Technical Recycling",
 7:"Industrial Sectors & Applied Cases",8:"Industrial Sectors & Applied Cases",9:"Industrial Sectors & Applied Cases",
 2:"Energy & Resource Systems",14:"Energy & Resource Systems",
 4:"Business, Policy & Governance",6:"Business, Policy & Governance",
 3:"Sustainability Framing & Society",5:"Sustainability Framing & Society"}
SF = "Sustainability Framing & Society"

def norm(s):
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"^https?://(dx\.)?doi\.org/","",regex=True))

req=["paper_topics_clean.csv","ce_papers_meta.csv","ce_authorships.csv",
     "ce_affiliations.csv","ce_institutions.csv"]
miss=[f for f in req if not Path(f).exists()]
if miss:
    print("FALTAN:",miss); sys.exit(1)
has_abstracts = Path("corpus_with_clusters.csv").exists()

print("Cargando y uniendo...")
topics=pd.read_csv("paper_topics_clean.csv",on_bad_lines="skip",dtype={"cluster":"Int64"})
topics=topics.dropna(subset=["cluster"]); topics["cluster"]=topics["cluster"].astype(int)
topics["mega_area"]=topics["cluster"].map(MEGA); topics=topics.dropna(subset=["mega_area"])
topics["doi_norm"]=norm(topics["doi"])

papers=pd.read_csv("ce_papers_meta.csv"); papers["doi_norm"]=norm(papers["doi"])
auth=pd.read_csv("ce_authorships.csv")
aff=pd.read_csv("ce_affiliations.csv")
inst=pd.read_csv("ce_institutions.csv")

t=topics.merge(papers[["openalex_id","doi_norm","type","is_oa","citations","year"]]
               .drop_duplicates("doi_norm"),on="doi_norm",how="inner").rename(columns={"openalex_id":"paper_id"})

# paper -> set de paises (unicos)
pa=auth[["paper_id","author_id"]].merge(aff,on="author_id",how="inner")
inst_c=inst[["openalex_id","country"]].rename(columns={"openalex_id":"institution_id"})
pac=pa.merge(inst_c,on="institution_id",how="inner")
pac=pac.dropna(subset=["country"]); pac=pac[pac["country"].astype(str).str.len()>0]

paper_countries=pac.groupby("paper_id")["country"].apply(lambda s:set(s.str.upper())).to_dict()
# anadir mega_area y metadata
meta=t.set_index("paper_id")[["mega_area","type","is_oa","citations","year"]].to_dict("index")

# ============================================================
print("\n"+"="*64)
print("H1 — Is Indonesia disproportionately represented in Sustainability Framing?")
print("="*64)
# conteo paper x pais por mega-area (mismo que el paper)
rows=[]
for pid,countries in paper_countries.items():
    if pid not in meta: continue
    ma=meta[pid]["mega_area"]
    for c in countries: rows.append((c,ma))
pc=pd.DataFrame(rows,columns=["country","mega_area"])

# distribucion de ID across mega-areas
id_dist=pc[pc.country=="ID"]["mega_area"].value_counts()
id_total=id_dist.sum()
print(f"\nIndonesia: {id_total} atribuciones paper-pais en total")
print("Distribucion de Indonesia por mega-area:")
for ma,n in id_dist.items():
    print(f"   {ma:38} {n:5}  ({100*n/id_total:4.1f}% del output ID)")

# comparar con la distribucion del corpus global
all_dist=pc["mega_area"].value_counts()
all_total=all_dist.sum()
print("\nIndice de concentracion (share ID en area / share corpus en area):")
print(f"   {'Mega-area':38} {'%ID':>7} {'%corpus':>9} {'ratio':>7}")
for ma in all_dist.index:
    sid=100*id_dist.get(ma,0)/id_total
    sall=100*all_dist[ma]/all_total
    ratio=sid/sall if sall else 0
    flag=" <-- sobre-representado" if ratio>1.3 else ""
    print(f"   {ma:38} {sid:6.1f}% {sall:8.1f}% {ratio:6.2f}{flag}")
print("\n  ratio>1 = Indonesia mas concentrada ahi que el corpus medio.")

# ============================================================
print("\n"+"="*64)
print("H3 — Single-country authorship (Indonesia) versus international collaboration")
print("="*64)
def collab_stats(country):
    n_solo=0; n_intl=0
    for pid,countries in paper_countries.items():
        if pid not in meta: continue
        if country in countries:
            if len(countries)==1: n_solo+=1
            else: n_intl+=1
    tot=n_solo+n_intl
    return n_solo,n_intl,tot
for cc in ["ID","CN","US","GB"]:
    solo,intl,tot=collab_stats(cc)
    if tot: print(f"   {cc}: {tot:5} papers | nacional-unico {100*solo/tot:4.1f}% | con co-pais internacional {100*intl/tot:4.1f}%")
print("\n  Si ID tiene MUCHO mas % nacional-unico que US/GB, su conteo paper-pais")
print("  esta menos 'repartido' -> ventaja artificial en el ranking por pais.")

# Solo dentro de Sustainability Framing
print("\n  -- Restringido a Sustainability Framing --")
def collab_in_area(country,area):
    n_solo=n_intl=0
    for pid,countries in paper_countries.items():
        if pid not in meta or meta[pid]["mega_area"]!=area: continue
        if country in countries:
            if len(countries)==1: n_solo+=1
            else: n_intl+=1
    tot=n_solo+n_intl; return n_solo,n_intl,tot
for cc in ["ID","CN","US","GB"]:
    solo,intl,tot=collab_in_area(cc,SF)
    if tot: print(f"   {cc}: {tot:5} | nacional-unico {100*solo/tot:4.1f}% | internacional {100*intl/tot:4.1f}%")

# ============================================================
print("\n"+"="*64)
print("H4 — Publication Type, Open Access Status, and Citation Characteristics")
print("="*64)
def type_oa_cites(country, area=None):
    types={}; oa=0; n=0; cites=[]
    for pid,countries in paper_countries.items():
        if pid not in meta: continue
        if area and meta[pid]["mega_area"]!=area: continue
        if country in countries:
            n+=1
            ty=str(meta[pid].get("type"))
            types[ty]=types.get(ty,0)+1
            if str(meta[pid].get("is_oa")).lower() in ("true","1"): oa+=1
            cc=meta[pid].get("citations")
            try: cites.append(float(cc))
            except: pass
    return n,types,oa,cites
print("\n  En Sustainability Framing:")
for cc in ["ID","CN","US","GB"]:
    n,types,oa,cites=type_oa_cites(cc,SF)
    if not n: continue
    med=np.median(cites) if cites else 0
    top_types=sorted(types.items(),key=lambda x:-x[1])[:3]
    print(f"   {cc}: n={n:4} | OA={100*oa/n:4.1f}% | citas_mediana={med:4.0f} | tipos={top_types}")
print("\n  Si ID tiene OA muy alto, citas medianas muy bajas, y mas 'proceedings',")
print("  apunta a output local/poco citado sobre-indexado por OpenAlex (artefacto).")

# ============================================================
if has_abstracts:
    print("\n"+"="*64)
    print("H2 — Abstract length and vocabulary characteristics of Indonesian publications in Sustainability Framing")
    print("="*64)
    cwc=pd.read_csv("corpus_with_clusters.csv")
    cwc["doi_norm"]=norm(cwc["doi"])
    cwc["abs_words"]=cwc["abstract"].fillna("").astype(str).str.split().apply(len)
    # mapear doi -> set paises
    doi_paper=t[["doi_norm","paper_id","mega_area"]]
    cwc2=cwc.merge(doi_paper,on="doi_norm",how="inner")
    def is_id(pid): return pid in paper_countries and "ID" in paper_countries[pid]
    sf=cwc2[cwc2.mega_area==SF].copy()
    sf["is_ID"]=sf["paper_id"].apply(is_id)
    id_w=sf[sf.is_ID]["abs_words"]; other_w=sf[~sf.is_ID]["abs_words"]
    print(f"\n  Abstracts en Sustainability Framing:")
    print(f"   Indonesia: n={len(id_w):4} | palabras_mediana={id_w.median():.0f} | media={id_w.mean():.0f}")
    print(f"   Resto:     n={len(other_w):4} | palabras_mediana={other_w.median():.0f} | media={other_w.mean():.0f}")
    # generic vocab proxy: % abstracts que mencionan 'sdg' o 'sustainable development'
    import re
    gen=re.compile(r"sustainable development|sdg|circular economy concept",re.I)
    sf["generic"]=sf["abstract"].fillna("").astype(str).str.contains(gen)
    print(f"\n   % con vocab generico (SDG/sust.dev/CE-concept):")
    print(f"   Indonesia: {100*sf[sf.is_ID]['generic'].mean():.1f}%")
    print(f"   Resto:     {100*sf[~sf.is_ID]['generic'].mean():.1f}%")
else:
    print("\n[H2 saltado: falta corpus_with_clusters.csv]")

print("\n"+"="*64)
print("LISTO. Usa estos numeros para el parrafo de cautela (Reviewer 1).")
print("="*64)
