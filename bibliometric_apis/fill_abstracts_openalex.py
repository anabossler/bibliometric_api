"""
OPENALEX ABSTRACTS COMPLETER
Completa abstracts faltantes usando OpenAlex API y actualiza Neo4j.

Usa variables del .env:
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
- OPENALEX_EMAIL (para polite pool; tb sirve CROSSREF_EMAIL como fallback)
- MAX_ABSTRACTS (0 = todos), OPENALEX_SLEEP

Actualiza:
- p.abstract (completa los abstracts vacíos)
- p.openalex_id
- p.language (solo si está vacía)
- p.abstract_source = 'openalex' (cuando añade abstract)
- p.openalex_checked = true (para marcar que ya se intentó)
"""

import os, time, requests
from typing import List, Tuple, Optional
from neo4j import GraphDatabase

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv no instalado. Instalando...")
    import subprocess
    subprocess.check_call(["pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "openalexalzheimer")

OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL") or os.getenv("CROSSREF_EMAIL") or "you@example.com"
MAX_ABSTRACTS  = int(os.getenv("MAX_ABSTRACTS", "0"))   # 0 = todos
SLEEP_SEC      = float(os.getenv("OPENALEX_SLEEP", "0.2"))

UA_HEADERS = {"User-Agent": f"neo4j-openalex/1.0 (mailto:{OPENALEX_EMAIL})"}

# ---------- Neo4j ----------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_dois_missing_abstract(limit: int = 0) -> List[str]:
    """
    DOIs cuyo abstract está vacío y aún no se ha verificado en OpenAlex.
    """
    cypher = """
    MATCH (p:Paper)
    WHERE coalesce(p.abstract, '') = ''
      AND coalesce(p.openalex_checked, false) = false
      AND p.doi IS NOT NULL
    RETURN p.doi AS doi
    """
    if limit and limit > 0:
        cypher += " LIMIT $limit"
    with driver.session(database=NEO4J_DATABASE) as s:
        res = s.run(cypher, limit=limit)
        return [r["doi"].lower() for r in res]

def update_abstract_batch(rows: List[Tuple[str, str, Optional[str], bool]]):
    """
    rows: lista de tuplas (doi, abstract_text, language, found_in_openalex)
    """
    if not rows:
        return
    
    cypher = """
    UNWIND $rows AS r
    MATCH (p:Paper {doi: r.doi})
    SET p.abstract = CASE 
            WHEN r.abst IS NOT NULL AND r.abst <> '' THEN r.abst
            ELSE p.abstract
        END,
        p.openalex_checked = true,
        p.abstract_source = CASE 
            WHEN r.abst IS NOT NULL AND r.abst <> '' THEN 'openalex'
            ELSE coalesce(p.abstract_source, 'none')
        END,
        p.language = coalesce(p.language, r.lang)
    """
    
    # Incluye todos los registros, incluso sin abstract (con cadena vacía)
    data = [{"doi": d, "abst": a or "", "lang": l} for d, a, l, _ in rows]
    
    with driver.session(database=NEO4J_DATABASE) as s:
        s.run(cypher, rows=data)
        print(f"✓ Batch actualizado: {len(data)} registros")

def set_openalex_id(doi: str, openalex_id: str):
    """Guardar el ID de OpenAlex para el paper"""
    if not openalex_id:
        return
    with driver.session(database=NEO4J_DATABASE) as s:
        s.run("MATCH (p:Paper {doi:$d}) SET p.openalex_id=$id", d=doi, id=openalex_id)

def get_stats():
    """Obtiene estadísticas actuales de la base de datos."""
    cypher = """
    MATCH (p:Paper)
    RETURN 
      count(*) as total,
      sum(CASE WHEN coalesce(p.abstract, '') <> '' THEN 1 ELSE 0 END) as con_abstract,
      sum(CASE WHEN coalesce(p.abstract, '') = '' THEN 1 ELSE 0 END) as sin_abstract,
      sum(CASE WHEN coalesce(p.openalex_checked, false) = true THEN 1 ELSE 0 END) as ya_verificados,
      sum(CASE WHEN p.abstract_source = 'openalex' THEN 1 ELSE 0 END) as abstracts_openalex,
      sum(CASE WHEN p.abstract_source = 'crossref' THEN 1 ELSE 0 END) as abstracts_crossref
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        result = s.run(cypher).single()
        return dict(result) if result else {}

# ---------- OpenAlex ----------
def reconstruct_from_inverted_index(ii: dict) -> str:
    """Convierte abstract_inverted_index (palabra -> posiciones) a texto plano."""
    try:
        if not ii:
            return ""
        max_pos = max((max(idxs) for idxs in ii.values() if idxs), default=-1)
        if max_pos < 0:
            return ""
        arr = [""] * (max_pos + 1)
        for word, idxs in ii.items():
            for pos in idxs:
                if 0 <= pos < len(arr):
                    arr[pos] = word
        return " ".join(w for w in arr if w).strip()
    except Exception:
        return ""

def fetch_openalex_work_by_doi(doi: str) -> dict:
    """
    Llama a OpenAlex para un DOI concreto, en polite pool (mailto + User-Agent).
    """
    # Endpoint directo por DOI
    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"mailto": OPENALEX_EMAIL}
    
    try:
        r = requests.get(url, headers=UA_HEADERS, params=params, timeout=40)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            # Fallback: endpoint de búsqueda por filtro
            r = requests.get(
                "https://api.openalex.org/works",
                headers=UA_HEADERS,
                params={
                    "filter": f"doi:{doi}", 
                    "select": "id,doi,abstract_inverted_index,language", 
                    "mailto": OPENALEX_EMAIL
                },
                timeout=40,
            )
            r.raise_for_status()
            j = r.json()
            items = j.get("results") or j.get("data") or []
            return items[0] if items else {}
        else:
            r.raise_for_status()
    except requests.RequestException:
        return {}

# ---------- Main ----------
def main():
    print("\n🔬 OPENALEX ABSTRACTS COMPLETER")
    print("="*50)
    
    # Mostrar estadísticas iniciales
    stats = get_stats()
    print("=== ESTADÍSTICAS INICIALES ===")
    print(f"Total papers: {stats.get('total', 0):,}")
    print(f"Con abstract: {stats.get('con_abstract', 0):,}")
    print(f"Sin abstract: {stats.get('sin_abstract', 0):,}")
    print(f"Ya verificados en OpenAlex: {stats.get('ya_verificados', 0):,}")
    print(f"Abstracts de OpenAlex: {stats.get('abstracts_openalex', 0):,}")
    print(f"Abstracts de Crossref: {stats.get('abstracts_crossref', 0):,}")
    print(f"Base de datos: {NEO4J_DATABASE}")
    print("=" * 35)
    
    dois = get_dois_missing_abstract(MAX_ABSTRACTS)
    print(f"\n📋 DOIs a procesar: {len(dois):,}")
    
    if not dois:
        print("✅ No hay DOIs para procesar. Todos los papers sin abstract ya fueron verificados en OpenAlex.")
        return

    # Confirmación del usuario
    if len(dois) > 1000:
        confirm = input(f"\n⚠️  Vas a procesar {len(dois):,} papers. ¿Continuar? (s/n): ")
        if not confirm.lower().startswith('s'):
            print("❌ Cancelado por el usuario")
            return

    print(f"\n🚀 Iniciando procesamiento...")
    
    batch, n_ok, n_fail = [], 0, 0
    start_time = time.time()
    
    for i, doi in enumerate(dois, 1):
        try:
            j = fetch_openalex_work_by_doi(doi)
            found_in_openalex = bool(j)
            
            if not j:
                # Marcar como verificado aunque no se encontró
                batch.append((doi, "", None, False))
                n_fail += 1
                print(f"[{i:,}/{len(dois):,}] ❌ no encontrado en OpenAlex: {doi}")
                time.sleep(SLEEP_SEC)
                continue

            openalex_id = j.get("id")
            if openalex_id:
                set_openalex_id(doi, openalex_id)

            ii   = j.get("abstract_inverted_index")
            lang = j.get("language")
            abst = reconstruct_from_inverted_index(ii) if ii else ""

            if abst:
                batch.append((doi, abst, lang, True))
                n_ok += 1
                print(f"[{i:,}/{len(dois):,}] ✅ abstract ({len(abst)} chars): {doi}")
            else:
                # Marcar como verificado aunque no tiene abstract
                batch.append((doi, "", lang, True))
                n_fail += 1
                print(f"[{i:,}/{len(dois):,}] ⚪ sin abstract en OpenAlex: {doi}")

            # Procesar batch cada 50 elementos para evitar problemas de memoria
            if len(batch) >= 50:
                update_abstract_batch(batch)
                batch = []
                
                # Mostrar progreso cada 500 elementos
                if i % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(dois) - i) / rate if rate > 0 else 0
                    print(f"\n📊 Progreso: {i:,}/{len(dois):,} ({i/len(dois)*100:.1f}%)")
                    print(f"⏱️  Velocidad: {rate:.1f} papers/sec, ETA: {eta/60:.1f} min")
                    print(f"✅ Abstracts encontrados: {n_ok:,} | ❌ Sin abstract: {n_fail:,}\n")
            
            time.sleep(SLEEP_SEC)

        except requests.HTTPError as e:
            n_fail += 1
            code = e.response.status_code if e.response is not None else "NA"
            print(f"[{i:,}/{len(dois):,}] HTTP {code}: {doi}")
            # Marcar como verificado aunque hubo error
            batch.append((doi, "", None, False))
            time.sleep(SLEEP_SEC)
        except Exception as e:
            n_fail += 1
            print(f"[{i:,}/{len(dois):,}] ERROR {type(e).__name__}: {e} - {doi}")
            # Marcar como verificado aunque hubo error
            batch.append((doi, "", None, False))
            time.sleep(SLEEP_SEC)

    # Procesar el último batch
    if batch:
        update_abstract_batch(batch)

    elapsed = time.time() - start_time
    
    print(f"\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"⏱️  Tiempo total: {elapsed/60:.1f} minutos")
    print(f"✅ Abstracts añadidos: {n_ok:,}")
    print(f"❌ Sin abstract: {n_fail:,}")
    print(f"📈 Tasa de éxito: {n_ok/(n_ok+n_fail)*100:.1f}%")
    
    # Mostrar estadísticas finales
    stats_final = get_stats()
    print("\n=== ESTADÍSTICAS FINALES ===")
    print(f"Total papers: {stats_final.get('total', 0):,}")
    print(f"Con abstract: {stats_final.get('con_abstract', 0):,}")
    print(f"Sin abstract: {stats_final.get('sin_abstract', 0):,}")
    print(f"Ya verificados en OpenAlex: {stats_final.get('ya_verificados', 0):,}")
    print(f"Abstracts de OpenAlex: {stats_final.get('abstracts_openalex', 0):,}")
    print(f"Abstracts de Crossref: {stats_final.get('abstracts_crossref', 0):,}")
    
    # Calcular mejora
    initial_abstracts = stats.get('con_abstract', 0)
    final_abstracts = stats_final.get('con_abstract', 0)
    improvement = final_abstracts - initial_abstracts
    
    if improvement > 0:
        total_papers = stats_final.get('total', 1)
        coverage_before = initial_abstracts / total_papers * 100
        coverage_after = final_abstracts / total_papers * 100
        
        print(f"\n📈 MEJORA OBTENIDA:")
        print(f"Cobertura antes: {coverage_before:.1f}%")
        print(f"Cobertura después: {coverage_after:.1f}%") 
        print(f"Mejora: +{improvement:,} abstracts (+{coverage_after-coverage_before:.1f}%)")
    
    print("=" * 31)

if __name__ == "__main__":
    main()