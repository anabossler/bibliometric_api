"""
Crea relaciones de grafo para análisis de conocimiento científico.

Relaciones que crea:
- CO_AUTHORED: Entre autores que han publicado juntos
- CO_CITED: Entre papers citados juntos en otros papers
- SIMILAR_TOPIC: Entre papers con temas similares
- CITED_TOGETHER: Entre autores cuyos trabajos se citan juntos
- COLLABORATED_WITH: Entre autores con co-autorías múltiples
- TOPIC_EXPERT: Entre autores y sus temas principales

Usa variables del .env:
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import os, time
from typing import Dict, List, Tuple
from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Config ----------
NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "openalexalzheimer")

# Parámetros de similitud
MIN_COAUTHORSHIP_STRENGTH = 2  # Mínimo papers juntos para relación fuerte
MIN_COCITATION_STRENGTH = 3    # Mínimo co-citaciones para relación
MIN_TOPIC_SIMILARITY = 0.3     # Similaridad mínima de temas (Jaccard)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class GraphRelationshipBuilder:
    def __init__(self):
        self.stats = {}
    
    def log_stats(self, relationship_type: str, count: int):
        """Registra estadísticas de relaciones creadas."""
        self.stats[relationship_type] = count
        print(f"✓ {relationship_type}: {count} relaciones creadas")
    
    def create_coauthorship_relations(self):
        """Crea relaciones CO_AUTHORED entre autores que han publicado juntos."""
        print("\n🤝 Creando relaciones de co-autoría...")
        
        cypher = """
        // Encuentra pares de autores que han colaborado
        MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE id(a1) < id(a2)  // Evita duplicados
        WITH a1, a2, count(p) as collaboration_count, collect(p.title) as papers
        WHERE collaboration_count >= 1
        
        // Crea la relación con estadísticas
        MERGE (a1)-[r:CO_AUTHORED]->(a2)
        SET r.collaboration_count = collaboration_count,
            r.strength = CASE 
                WHEN collaboration_count >= $min_strength THEN 'strong'
                WHEN collaboration_count >= 2 THEN 'medium' 
                ELSE 'weak' 
            END,
            r.papers = papers[0..5],  // Primeros 5 papers
            r.created_at = datetime()
        
        RETURN count(r) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, min_strength=MIN_COAUTHORSHIP_STRENGTH)
            count = result.single()["relations_created"]
            self.log_stats("CO_AUTHORED", count)
    
    def create_cocitation_relations(self):
        """Crea relaciones CO_CITED entre papers citados juntos."""
        print("\n📚 Creando relaciones de co-citación...")
        
        cypher = """
        // Encuentra papers que son citados juntos
        MATCH (citing:Paper)-[:CITES]->(p1:Paper)
        MATCH (citing)-[:CITES]->(p2:Paper)
        WHERE id(p1) < id(p2)  // Evita duplicados
        WITH p1, p2, count(citing) as cocitation_count, collect(citing.title) as citing_papers
        WHERE cocitation_count >= $min_citations
        
        // Crea la relación
        MERGE (p1)-[r:CO_CITED]->(p2)
        SET r.cocitation_count = cocitation_count,
            r.strength = CASE 
                WHEN cocitation_count >= 10 THEN 'strong'
                WHEN cocitation_count >= 5 THEN 'medium'
                ELSE 'weak'
            END,
            r.citing_papers = citing_papers[0..3],  // Primeros 3 papers citantes
            r.created_at = datetime()
        
        RETURN count(r) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, min_citations=MIN_COCITATION_STRENGTH)
            count = result.single()["relations_created"]
            self.log_stats("CO_CITED", count)
    
    def create_topic_similarity_relations(self):
        """Crea relaciones SIMILAR_TOPIC entre papers con temas similares."""
        print("\n🏷️ Creando relaciones de similitud temática...")
        
        cypher = """
        // Encuentra papers con temas similares (usando subjects de Crossref)
        MATCH (p1:Paper), (p2:Paper)
        WHERE id(p1) < id(p2) 
          AND size(p1.crossref_subjects) > 0 
          AND size(p2.crossref_subjects) > 0
        
        // Calcula similitud Jaccard
        WITH p1, p2,
             [x IN p1.crossref_subjects WHERE x IN p2.crossref_subjects] as common,
             p1.crossref_subjects + [x IN p2.crossref_subjects WHERE NOT x IN p1.crossref_subjects] as union_subjects
        
        WITH p1, p2, common, union_subjects,
             toFloat(size(common)) / size(union_subjects) as jaccard_similarity
        
        WHERE jaccard_similarity >= $min_similarity
        
        // Crea la relación
        MERGE (p1)-[r:SIMILAR_TOPIC]->(p2)
        SET r.jaccard_similarity = jaccard_similarity,
            r.common_subjects = common,
            r.strength = CASE 
                WHEN jaccard_similarity >= 0.7 THEN 'strong'
                WHEN jaccard_similarity >= 0.5 THEN 'medium'
                ELSE 'weak'
            END,
            r.created_at = datetime()
        
        RETURN count(r) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, min_similarity=MIN_TOPIC_SIMILARITY)
            count = result.single()["relations_created"]
            self.log_stats("SIMILAR_TOPIC", count)
    
    def create_author_citation_relations(self):
        """Crea relaciones CITED_TOGETHER entre autores cuyos trabajos se citan juntos."""
        print("\n🔗 Creando relaciones de citación conjunta entre autores...")
        
        cypher = """
        // Encuentra autores cuyos papers son citados juntos
        MATCH (a1:Author)-[:AUTHORED]->(p1:Paper)-[:CO_CITED]-(p2:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE id(a1) < id(a2)
        WITH a1, a2, count(*) as citation_strength, 
             collect({p1: p1.title, p2: p2.title}) as cited_pairs
        WHERE citation_strength >= 2
        
        // Crea la relación
        MERGE (a1)-[r:CITED_TOGETHER]->(a2)
        SET r.citation_strength = citation_strength,
            r.cited_pairs = cited_pairs[0..3],
            r.strength = CASE 
                WHEN citation_strength >= 10 THEN 'strong'
                WHEN citation_strength >= 5 THEN 'medium'
                ELSE 'weak'
            END,
            r.created_at = datetime()
        
        RETURN count(r) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher)
            count = result.single()["relations_created"]
            self.log_stats("CITED_TOGETHER", count)
    
    def create_strong_collaboration_relations(self):
        """Crea relaciones COLLABORATED_WITH para colaboraciones frecuentes."""
        print("\n💪 Creando relaciones de colaboración fuerte...")
        
        cypher = """
        // Identifica colaboraciones fuertes (múltiples papers juntos)
        MATCH (a1:Author)-[r:CO_AUTHORED]->(a2:Author)
        WHERE r.collaboration_count >= $min_strength
        
        // Crea relación de colaboración fuerte
        MERGE (a1)-[cr:COLLABORATED_WITH]->(a2)
        SET cr.collaboration_count = r.collaboration_count,
            cr.papers = r.papers,
            cr.relationship_strength = r.strength,
            cr.created_at = datetime()
        
        RETURN count(cr) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, min_strength=MIN_COAUTHORSHIP_STRENGTH)
            count = result.single()["relations_created"]
            self.log_stats("COLLABORATED_WITH", count)
    
    def create_topic_expert_relations(self):
        """Crea relaciones TOPIC_EXPERT entre autores y sus temas principales."""
        print("\n🎓 Creando relaciones de expertise temática...")
        
        cypher = """
        // Encuentra los temas principales de cada autor
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)
        WHERE size(p.crossref_subjects) > 0
        UNWIND p.crossref_subjects as subject
        WITH a, subject, count(*) as paper_count
        WHERE paper_count >= 3  // Al menos 3 papers en el tema
        
        // Crear nodos Topic si no existen
        MERGE (t:Topic {name: subject})
        ON CREATE SET t.created_at = datetime()
        
        // Crear relación TOPIC_EXPERT
        MERGE (a)-[r:TOPIC_EXPERT]->(t)
        SET r.paper_count = paper_count,
            r.expertise_level = CASE 
                WHEN paper_count >= 10 THEN 'expert'
                WHEN paper_count >= 5 THEN 'experienced'
                ELSE 'familiar'
            END,
            r.created_at = datetime()
        
        RETURN count(r) as relations_created
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher)
            count = result.single()["relations_created"]
            self.log_stats("TOPIC_EXPERT", count)
    
    def create_temporal_relations(self):
        """Crea relaciones temporales para análisis de evolución."""
        print("\n⏰ Creando relaciones temporales...")
        
        cypher = """
        // Relaciona autores con su evolución temporal
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)
        WHERE p.pub_year IS NOT NULL
        WITH a, min(p.pub_year) as first_year, max(p.pub_year) as last_year, 
             count(p) as total_papers, collect(p.pub_year) as years
        WHERE total_papers >= 2 AND last_year > first_year
        
        SET a.career_start = first_year,
            a.career_latest = last_year,
            a.career_span = last_year - first_year,
            a.total_papers = total_papers,
            a.publication_years = years,
            a.productivity = toFloat(total_papers) / (last_year - first_year + 1)
        
        RETURN count(a) as authors_updated
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher)
            count = result.single()["authors_updated"]
            self.log_stats("TEMPORAL_PROPERTIES", count)
    
    def create_influence_metrics(self):
        """Calcula métricas de influencia y centralidad."""
        print("\n📊 Calculando métricas de influencia...")
        
        # H-index simplificado para autores
        cypher_hindex = """
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)
        WITH a, p.is_referenced_by_count as citations
        ORDER BY citations DESC
        WITH a, collect(citations) as citation_list
        WITH a, citation_list, range(1, size(citation_list)) as indices
        UNWIND indices as i
        WITH a, citation_list[i-1] as citations, i
        WHERE citations >= i
        WITH a, max(i) as h_index
        SET a.h_index = h_index
        RETURN count(a) as authors_updated
        """
        
        # Grado de colaboración
        cypher_collab = """
        MATCH (a:Author)-[r:CO_AUTHORED]-()
        WITH a, count(r) as collaboration_degree, sum(r.collaboration_count) as total_collaborations
        SET a.collaboration_degree = collaboration_degree,
            a.total_collaborations = total_collaborations
        RETURN count(a) as authors_updated
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result1 = session.run(cypher_hindex)
            count1 = result1.single()["authors_updated"]
            
            result2 = session.run(cypher_collab)
            count2 = result2.single()["authors_updated"]
            
            self.log_stats("H_INDEX_CALCULATED", count1)
            self.log_stats("COLLABORATION_METRICS", count2)
    
    def get_graph_statistics(self):
        """Obtiene estadísticas del grafo completo."""
        print("\n📈 Estadísticas del grafo:")
        
        queries = {
            "Nodos": "MATCH (n) RETURN count(n) as count",
            "Papers": "MATCH (p:Paper) RETURN count(p) as count",
            "Autores": "MATCH (a:Author) RETURN count(a) as count", 
            "Temas": "MATCH (t:Topic) RETURN count(t) as count",
            "Relaciones": "MATCH ()-[r]->() RETURN count(r) as count",
            "Co-autorías": "MATCH ()-[r:CO_AUTHORED]->() RETURN count(r) as count",
            "Co-citaciones": "MATCH ()-[r:CO_CITED]->() RETURN count(r) as count",
            "Similaridad temática": "MATCH ()-[r:SIMILAR_TOPIC]->() RETURN count(r) as count"
        }
        
        stats = {}
        with driver.session(database=NEO4J_DATABASE) as session:
            for name, query in queries.items():
                result = session.run(query)
                count = result.single()["count"]
                stats[name] = count
                print(f"  {name}: {count:,}")
        
        return stats

def main():
    print("🔄 Construyendo relaciones de grafo de conocimiento científico...")
    print(f"📍 Base de datos: {NEO4J_DATABASE}")
    print("=" * 60)
    
    builder = GraphRelationshipBuilder()
    start_time = time.time()
    
    try:
        # Crear relaciones paso a paso
        builder.create_coauthorship_relations()
        builder.create_cocitation_relations()
        builder.create_topic_similarity_relations()
        builder.create_author_citation_relations()
        builder.create_strong_collaboration_relations()
        builder.create_topic_expert_relations()
        builder.create_temporal_relations()
        builder.create_influence_metrics()
        
        # Estadísticas finales
        print("\n" + "=" * 60)
        stats = builder.get_graph_statistics()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {elapsed:.2f} segundos")
        print(f"🎯 Total relaciones creadas: {sum(builder.stats.values()):,}")
        
        print("\n✅ Construcción del grafo completada!")
        
        print("\n🔍 Consultas de ejemplo para explorar:")
        print("// Encontrar colaboradores frecuentes de un autor:")
        print("MATCH (a:Author {family: 'Smith'})-[r:COLLABORATED_WITH]->(colleague)")
        print("RETURN colleague.full_name, r.collaboration_count ORDER BY r.collaboration_count DESC")
        
        print("\n// Papers más co-citados:")
        print("MATCH (p1:Paper)-[r:CO_CITED]->(p2:Paper)")
        print("RETURN p1.title, p2.title, r.cocitation_count ORDER BY r.cocitation_count DESC LIMIT 10")
        
        print("\n// Expertos en un tema:")
        print("MATCH (a:Author)-[r:TOPIC_EXPERT]->(t:Topic {name: 'alzheimer'})")
        print("RETURN a.full_name, r.paper_count, r.expertise_level ORDER BY r.paper_count DESC")
        
    except Exception as e:
        print(f"❌ Error durante la construcción: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()