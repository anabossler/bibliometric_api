"""Clasifica los terminos Objective y Technique existentes en el grafo.
NO toca el grafo. Salida: CSV auditable term -> categoria.
Uso: export NEO4J_PW=...; python classify_tmo_terms.py
"""
import os, json, re, time, hashlib
from datetime import datetime, timezone
import pandas as pd, requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
KEY = os.environ["OPENROUTER_API_KEY"]
assert KEY.startswith("sk-or-"), "API key invalida en .env"
MODEL, BATCH = "google/gemini-2.5-flash-lite", 40
URI, DB = "bolt://localhost:7691", "circulareconomy"

drv = GraphDatabase.driver(URI, auth=("neo4j", os.environ["NEO4J_PW"]))
Q = {"objective": "MATCH (p:Paper)-[:PURSUES_OBJECTIVE]->(o:Objective) RETURN o.name AS term, count(p) AS papers ORDER BY papers DESC",
     "technique": "MATCH (p:Paper)-[:USES_TECHNIQUE]->(t:Technique) RETURN t.name AS term, count(p) AS papers ORDER BY papers DESC"}

def run(kind):
    prompt = open(f"prompt_classify_{kind}.txt").read()
    psha = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    out = f"tmo_{kind}_classified.csv"
    with drv.session(database=DB) as s:
        terms = pd.DataFrame([dict(r) for r in s.run(Q[kind])]).dropna(subset=["term"])
    print(f"[{kind}] {len(terms):,} terminos unicos, {terms.papers.sum():,} papers-rel")

    done = {}
    if os.path.exists(out):
        d = pd.read_csv(out); done = dict(zip(d.term, d.to_dict("records")))
        print(f"  reanudando: {len(done)} ya hechos")
    todo = [t for t in terms.term if t not in done]

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        for attempt in range(4):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={"model": MODEL, "temperature": 0, "max_tokens": 2500,
                          "messages":[{"role":"system","content":prompt},
                                      {"role":"user","content":json.dumps(chunk)}]},
                    timeout=90)
                if r.status_code in (401,403): raise SystemExit(f"AUTH: {r.text[:150]}")
                r.raise_for_status()
                txt = re.sub(r"```json|```","",r.json()["choices"][0]["message"]["content"]).strip()
                res = json.loads(txt)["results"]
                for x in res:
                    t = x.get("term")
                    if t: done[t] = {"term": t,
                        "category": x.get("primary_domain") or x.get("category"),
                        "secondary": "|".join(x.get("secondary_domains") or x.get("secondary") or []),
                        "model": MODEL, "prompt_sha": psha,
                        "created_at": datetime.now(timezone.utc).isoformat()}
                break
            except SystemExit: raise
            except Exception as e:
                if attempt == 3: print(f"  !! batch {i} fallo: {str(e)[:90]}")
                time.sleep(2**attempt)
        df = terms.merge(pd.DataFrame(done.values()), on="term", how="left")
        df.to_csv(out, index=False)
        print(f"  {min(i+BATCH,len(todo))}/{len(todo)}", end="\r")

    df = terms.merge(pd.DataFrame(done.values()), on="term", how="left")
    df.to_csv(out, index=False)
    print(f"\n[{kind}] -> {out}")
    # cobertura ponderada POR PAPERS, no por terminos unicos
    w = df.groupby("category").papers.sum().sort_values(ascending=False)
    print((w / df.papers.sum() * 100).round(1).to_string())
    print(f"  sin clasificar: {df.category.isna().sum()} terminos")

for k in ("technique", "objective"): run(k)
drv.close()
