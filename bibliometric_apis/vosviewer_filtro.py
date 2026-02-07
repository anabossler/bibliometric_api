"""
Analiza intersecciones entre términos clave en Neo4j
y genera CSVs separados para VOSviewer
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "circulareconomy")

OUT_DIR = "term_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Términos a analizar
TERMS = {
    "ce": "circular economy",
    "zw": "zero waste",
    "sust": "sustainability",
    "gc": "green chemistry"
}

def connect():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    with driver.session(database=NEO4J_DB) as s:
        s.run("RETURN 1").consume()
    return driver

def analyze_intersections(driver):
    """Analiza cuántos papers tienen cada combinación de términos"""
    
    print("\n" + "="*60)
    print("ANÁLISIS DE INTERSECCIONES")
    print("="*60)
    
    results = []
    
    with driver.session(database=NEO4J_DB) as session:
        # Query para todas las combinaciones
        query = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL
        WITH p,
             CASE WHEN (toLower(p.title) CONTAINS 'circular economy' OR 
                        toLower(p.abstract) CONTAINS 'circular economy') 
                  THEN 1 ELSE 0 END AS has_ce,
             CASE WHEN (toLower(p.title) CONTAINS 'zero waste' OR 
                        toLower(p.abstract) CONTAINS 'zero waste') 
                  THEN 1 ELSE 0 END AS has_zw,
             CASE WHEN (toLower(p.title) CONTAINS 'sustainability' OR 
                        toLower(p.abstract) CONTAINS 'sustainability' OR
                        toLower(p.title) CONTAINS 'sustainable' OR
                        toLower(p.abstract) CONTAINS 'sustainable') 
                  THEN 1 ELSE 0 END AS has_sust,
             CASE WHEN (toLower(p.title) CONTAINS 'green chemistry' OR 
                        toLower(p.abstract) CONTAINS 'green chemistry') 
                  THEN 1 ELSE 0 END AS has_gc
        
        WHERE has_ce = 1 OR has_zw = 1 OR has_sust = 1 OR has_gc = 1
        
        RETURN 
            sum(has_ce) AS total_ce,
            sum(has_zw) AS total_zw,
            sum(has_sust) AS total_sust,
            sum(has_gc) AS total_gc,
            sum(has_ce * has_zw) AS ce_and_zw,
            sum(has_ce * has_sust) AS ce_and_sust,
            sum(has_ce * has_gc) AS ce_and_gc,
            sum(has_zw * has_sust) AS zw_and_sust,
            sum(has_zw * has_gc) AS zw_and_gc,
            sum(has_sust * has_gc) AS sust_and_gc,
            sum(has_ce * has_zw * has_sust) AS ce_zw_sust,
            sum(has_ce * has_zw * has_gc) AS ce_zw_gc,
            sum(has_ce * has_sust * has_gc) AS ce_sust_gc,
            sum(has_zw * has_sust * has_gc) AS zw_sust_gc,
            sum(has_ce * has_zw * has_sust * has_gc) AS all_four
        """
        
        result = session.run(query).single()
        
        # Términos individuales
        print("\n📊 TÉRMINOS INDIVIDUALES:")
        print(f"  • Circular Economy:  {result['total_ce']:,}")
        print(f"  • Zero Waste:        {result['total_zw']:,}")
        print(f"  • Sustainability:    {result['total_sust']:,}")
        print(f"  • Green Chemistry:   {result['total_gc']:,}")
        
        # Intersecciones de 2
        print("\n🔗 INTERSECCIONES (2 términos):")
        print(f"  • CE + ZW:           {result['ce_and_zw']:,}")
        print(f"  • CE + Sust:         {result['ce_and_sust']:,}")
        print(f"  • CE + GC:           {result['ce_and_gc']:,}")
        print(f"  • ZW + Sust:         {result['zw_and_sust']:,}")
        print(f"  • ZW + GC:           {result['zw_and_gc']:,}")
        print(f"  • Sust + GC:         {result['sust_and_gc']:,}")
        
        # Intersecciones de 3
        print("\n🔗🔗 INTERSECCIONES (3 términos):")
        print(f"  • CE + ZW + Sust:    {result['ce_zw_sust']:,}")
        print(f"  • CE + ZW + GC:      {result['ce_zw_gc']:,}")
        print(f"  • CE + Sust + GC:    {result['ce_sust_gc']:,}")
        print(f"  • ZW + Sust + GC:    {result['zw_sust_gc']:,}")
        
        # Intersección de 4
        print("\n🔗🔗🔗 INTERSECCIÓN (4 términos):")
        print(f"  • CE + ZW + Sust + GC: {result['all_four']:,}")
        
        return result

def export_intersection_network(driver, terms_list, output_name):
    """
    Exporta red de co-ocurrencia de términos específicos
    
    Args:
        terms_list: lista de términos a incluir (ej: ["circular economy", "sustainability"])
        output_name: nombre del archivo de salida
    """
    
    # Construir filtro WHERE
    conditions = []
    for term in terms_list:
        conditions.append(
            f"(toLower(p.title) CONTAINS '{term.lower()}' OR toLower(p.abstract) CONTAINS '{term.lower()}')"
        )
    
    filter_clause = " OR ".join(conditions)
    
    query = f"""
    MATCH (p:Paper)
    WHERE p.title IS NOT NULL
      AND ({filter_clause})
    
    OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
    OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
    
    WITH p, j,
         collect(DISTINCT a.display_name) AS authors,
         collect(DISTINCT c.display_name) AS concepts
    
    ORDER BY p.cited_by_count DESC
    LIMIT 10000
    
    RETURN
      p.openalex_id AS openalex_id,
      p.doi AS doi,
      p.title AS title,
      p.publication_year AS year,
      p.cited_by_count AS cited_by,
      p.abstract AS abstract,
      j.display_name AS journal,
      authors,
      concepts
    """
    
    print(f"\n📤 Exportando: {output_name}")
    print(f"   Términos: {terms_list}")
    
    with driver.session(database=NEO4J_DB) as session:
        results = session.run(query).data()
    
    if not results:
        print(f"   ❌ No se encontraron papers")
        return
    
    # Convertir a CSV formato Scopus
    rows = []
    for r in results:
        authors_str = "; ".join(r["authors"] or [])
        concepts_str = "; ".join(r["concepts"] or [])
        
        rows.append({
            "Authors": authors_str,
            "Author(s) ID": "",
            "Title": r["title"] or "",
            "Year": r["year"] or "",
            "Source title": r["journal"] or "",
            "Abstract": r["abstract"] or "",
            "Cited by": r["cited_by"] or 0,
            "DOI": r["doi"] or "",
            "Author Keywords": concepts_str,
            "EID": r["openalex_id"] or ""
        })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUT_DIR, f"{output_name}.csv")
    df.to_csv(output_path, index=False)
    
    print(f"   ✅ Exportados {len(rows):,} papers → {output_path}")

def main():
    print("\n🔬 ANÁLISIS DE TÉRMINOS CLAVE")
    
    driver = connect()
    print(f"✓ Conectado a: {NEO4J_DB}")
    
    # Analizar intersecciones
    analyze_intersections(driver)
    
    print("\n" + "="*60)
    print("GENERANDO EXPORTS ESPECÍFICOS")
    print("="*60)
    
    # 1. Red completa (4 términos)
    export_intersection_network(
        driver,
        ["circular economy", "zero waste", "sustainability", "green chemistry"],
        "network_4terms_all"
    )
    
    # 2. Solo CE + otros
    export_intersection_network(
        driver,
        ["circular economy", "zero waste"],
        "network_ce_zw"
    )
    
    export_intersection_network(
        driver,
        ["circular economy", "sustainability"],
        "network_ce_sust"
    )
    
    export_intersection_network(
        driver,
        ["circular economy", "green chemistry"],
        "network_ce_gc"
    )
    
    # 3. Intersección triple
    export_intersection_network(
        driver,
        ["circular economy", "sustainability", "zero waste"],
        "network_ce_sust_zw"
    )
    
    driver.close()
    
    print("\n✅ Análisis completado!")
    print(f"   Archivos en: {OUT_DIR}/")

if __name__ == "__main__":
    main()