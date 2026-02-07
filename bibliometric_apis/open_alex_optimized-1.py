"""
OPENALEX PRIMARY IMPORTER
Script completo para importar datos desde OpenAlex API como fuente primaria:
- Papers con abstracts completos, conceptos, keywords
- Autores con afiliaciones detalladas e instituciones
- Referencias bibliográficas 
- Journals y publishers
- Países e instituciones con coordenadas
- Conceptos y áreas de investigación
- Procesamiento simultáneo en batches
"""

import os
import re
import time
import json
import hashlib
import requests
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase
from datetime import datetime
import logging

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

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openalex_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración desde variables de entorno
class Config:
    def __init__(self):
        # Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "openalexalzheimer")
        
        # OpenAlex
        self.openalex_email = os.getenv("OPENALEX_EMAIL") or os.getenv("CROSSREF_EMAIL") or "your@email.com"
        self.max_papers_import = int(os.getenv("MAX_PAPERS_IMPORT", "10000"))
        self.batch_size_import = int(os.getenv("BATCH_SIZE_IMPORT", "50"))
        self.sleep_openalex = float(os.getenv("OPENALEX_SLEEP", "0.1"))
        
        # Filtros
        self.from_year = os.getenv("FROM_YEAR")
        self.to_year = os.getenv("TO_YEAR")
        
        # Directorio de datos
        self.data_dir = os.getenv("DATA_DIR", "./data")
        os.makedirs(self.data_dir, exist_ok=True)

config = Config()

class OpenAlexImporter:
    def __init__(self):
        self.driver = None
        self.headers = {"User-Agent": f"openalex-importer/1.0 (mailto:{config.openalex_email})"}
        
    def connect_to_neo4j(self):
        """Conectar a Neo4j"""
        try:
            self.driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"✓ Conexión a Neo4j exitosa - DB: {config.neo4j_database}")
            return True
        except Exception as e:
            logger.error(f"✗ Error conectando a Neo4j: {e}")
            return False
    
    def clear_database(self):
        """Limpiar base de datos completamente"""
        logger.info("🗑️  Limpiando base de datos...")
        with self.driver.session(database=config.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✓ Base de datos limpiada")
    
    def create_constraints_and_indexes(self):
        """Crear constraints e índices optimizados para OpenAlex"""
        logger.info("🔧 Creando constraints e índices...")
        
        with self.driver.session(database=config.neo4j_database) as session:
            # Constraints - Solo en campos que SIEMPRE tienen valores únicos
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalex_id IS UNIQUE",
                # DOI no es constraint porque muchos papers no tienen DOI (doi = '')
                # "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journal) REQUIRE j.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.code IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Funder) REQUIRE f.openalex_id IS UNIQUE"
            ]
            
            # Índices - Para búsquedas rápidas (pueden tener duplicados)
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.publication_year)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.cited_by_count)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi)",  # Índice SÍ, constraint NO
                "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.display_name)",
                "CREATE INDEX IF NOT EXISTS FOR (i:Institution) ON (i.display_name)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.display_name)"
            ]
            
            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(f"Error ejecutando: {stmt[:50]}... - {e}")
        
        logger.info("✓ Constraints e índices creados")
    
    def build_filter_string(self, query: str) -> str:
        """Construir filtros para OpenAlex"""
        filters = []
        
        # Filtro de búsqueda principal
        if query:
            filters.append(f"default.search:{query}")
        
        # Filtros de año
        if config.from_year:
            filters.append(f"from_publication_date:{config.from_year}-01-01")
        if config.to_year:
            filters.append(f"to_publication_date:{config.to_year}-12-31")
        
        # Solo artículos de journal por defecto
        filters.append("type:article")
        
        return ",".join(filters)
    
    def search_and_process_openalex(self, query: str):
        """Buscar y procesar papers en OpenAlex con procesamiento simultáneo"""
        logger.info(f"🔍 Buscando y procesando en OpenAlex: '{query}'")
        
        url = "https://api.openalex.org/works"
        filter_str = self.build_filter_string(query)
        
        params = {
            "filter": filter_str,
            "per_page": 200,
            "cursor": "*",
            "mailto": config.openalex_email,
            "select": "id,doi,title,display_name,publication_year,publication_date,type,abstract_inverted_index,authorships,primary_location,locations,concepts,referenced_works,related_works,cited_by_count,language"
        }
        
        batch_buffer = []
        fetched = 0
        total_imported = 0
        max_papers = config.max_papers_import if config.max_papers_import > 0 else float('inf')
        BATCH_SIZE = 500  # Procesar cada 500 papers
        
        while fetched < max_papers:
            logger.info(f"📡 Request OpenAlex - fetched: {fetched}, imported: {total_imported}")
            
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=45)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    logger.info("No hay más elementos")
                    break
                
                # Añadir items al buffer
                for item in results:
                    batch_buffer.append(item)
                    fetched += 1
                    
                    # Procesar e importar cuando el buffer llegue a 500
                    if len(batch_buffer) >= BATCH_SIZE:
                        imported_count = self._process_and_import_batch(batch_buffer, total_imported + 1)
                        total_imported += imported_count
                        batch_buffer = []  # Limpiar buffer
                        
                        # Mostrar estadísticas actuales
                        stats = self.get_database_stats()
                        logger.info(f"✅ BATCH COMPLETADO - Papers en DB: {stats.get('Papers', 0)}")
                    
                    if fetched >= max_papers:
                        break
                
                # Obtener siguiente página
                next_cursor = data.get("meta", {}).get("next_cursor")
                if not next_cursor:
                    logger.info("No hay más páginas")
                    break
                
                params["cursor"] = next_cursor
                time.sleep(config.sleep_openalex)
                
            except requests.RequestException as e:
                logger.error(f"Error en request: {e}")
                time.sleep(5)
                continue
        
        # Procesar último batch si queda algo
        if batch_buffer:
            imported_count = self._process_and_import_batch(batch_buffer, total_imported + 1)
            total_imported += imported_count
        
        logger.info(f"✓ PROCESAMIENTO COMPLETO - Total importados: {total_imported}")
        return total_imported
    
    def _process_and_import_batch(self, batch_data: List[Dict], start_num: int) -> int:
        """Procesar e importar un batch de papers"""
        logger.info(f"🔄 Procesando batch de {len(batch_data)} papers (#{start_num}-{start_num + len(batch_data) - 1})")
        
        processed_papers = []
        for i, paper_data in enumerate(batch_data):
            paper = self.process_paper(paper_data)
            if paper:
                processed_papers.append(paper)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Procesados {i + 1}/{len(batch_data)} papers del batch")
        
        logger.info(f"✓ {len(processed_papers)} papers procesados correctamente")
        
        # Importar inmediatamente a Neo4j
        if processed_papers:
            self.import_to_neo4j(processed_papers)
            logger.info(f"✅ Batch importado a Neo4j: {len(processed_papers)} papers")
        
        return len(processed_papers)
    
    def reconstruct_abstract(self, inverted_index: Dict) -> str:
        """Reconstruir abstract desde inverted index de OpenAlex"""
        try:
            if not inverted_index:
                return ""
            
            # Encontrar la posición máxima
            max_pos = 0
            for positions in inverted_index.values():
                if positions:
                    max_pos = max(max_pos, max(positions))
            
            if max_pos == 0:
                return ""
            
            # Crear array de palabras
            words = [""] * (max_pos + 1)
            for word, positions in inverted_index.items():
                for pos in positions:
                    if 0 <= pos <= max_pos:
                        words[pos] = word
            
            # Unir palabras no vacías
            return " ".join(word for word in words if word).strip()
            
        except Exception as e:
            logger.warning(f"Error reconstruyendo abstract: {e}")
            return ""
    
    def extract_authorships(self, authorships: List[Dict]) -> List[Dict]:
        """Extraer información de autorías"""
        if not authorships:
            return []
        
        authors = []
        for authorship in authorships:
            author_info = authorship.get("author", {})
            institutions = authorship.get("institutions", [])
            
            # Información del autor
            author_data = {
                "openalex_id": author_info.get("id", ""),
                "display_name": author_info.get("display_name", ""),
                "orcid": author_info.get("orcid"),
                "author_position": authorship.get("author_position"),
                "raw_author_name": authorship.get("raw_author_name"),
                "institutions": []
            }
            
            # Instituciones del autor
            for inst in institutions:
                if inst:
                    inst_data = {
                        "openalex_id": inst.get("id", ""),
                        "display_name": inst.get("display_name", ""),
                        "country_code": inst.get("country_code"),
                        "type": inst.get("type"),
                        "ror": inst.get("ror")
                    }
                    author_data["institutions"].append(inst_data)
            
            authors.append(author_data)
        
        return authors
    
    def extract_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Extraer conceptos de investigación"""
        if not concepts:
            return []
        
        concept_list = []
        for concept in concepts:
            concept_data = {
                "openalex_id": concept.get("id", ""),
                "display_name": concept.get("display_name", ""),
                "level": concept.get("level", 0),
                "score": concept.get("score", 0.0),
                "wikidata": concept.get("wikidata")
            }
            concept_list.append(concept_data)
        
        return concept_list
    
    def extract_location_info(self, locations: List[Dict]) -> Dict:
        """Extraer información de ubicación/journal"""
        if not locations:
            return {}
        
        # Usar primary_location si está disponible
        primary = locations[0] if locations else {}
        
        source = primary.get("source", {})
        if not source:
            return {}
        
        return {
            "journal_openalex_id": source.get("id", ""),
            "journal_name": source.get("display_name", ""),
            "journal_issn_l": source.get("issn_l"),
            "journal_type": source.get("type"),
            "publisher": source.get("host_organization_name"),
            "is_oa": primary.get("is_oa", False),
            "oa_date": primary.get("oa_date"),
            "oa_url": primary.get("oa_url")
        }
    
    def process_paper(self, paper_data: Dict) -> Optional[Dict]:
        """Procesar un paper individual de OpenAlex"""
        openalex_id = paper_data.get("id", "")
        if not openalex_id:
            return None
        
        # Datos básicos
        doi = paper_data.get("doi", "").replace("https://doi.org/", "").lower() if paper_data.get("doi") else ""
        title = paper_data.get("title") or paper_data.get("display_name", "")
        
        # Reconstruir abstract
        abstract = self.reconstruct_abstract(paper_data.get("abstract_inverted_index", {}))
        
        # Fechas y años
        publication_year = paper_data.get("publication_year")
        publication_date = paper_data.get("publication_date")
        
        # Métricas
        cited_by_count = paper_data.get("cited_by_count", 0)
        paper_type = paper_data.get("type", "")
        language = paper_data.get("language")
        
        # Datos complejos
        authorships = self.extract_authorships(paper_data.get("authorships", []))
        concepts = self.extract_concepts(paper_data.get("concepts", []))
        location_info = self.extract_location_info(paper_data.get("locations", []))
        
        # Referencias
        referenced_works = paper_data.get("referenced_works", [])
        related_works = paper_data.get("related_works", [])
        
        return {
            "openalex_id": openalex_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "publication_year": publication_year,
            "publication_date": publication_date,
            "type": paper_type,
            "language": language,
            "cited_by_count": cited_by_count,
            "authorships": authorships,
            "concepts": concepts,
            "location_info": location_info,
            "referenced_works": referenced_works,
            "related_works": related_works,
            "updated_at": datetime.now().isoformat()
        }
    
    def import_to_neo4j(self, papers: List[Dict]):
        """Importar papers a Neo4j con todas las relaciones"""
        logger.info(f"📥 Importando {len(papers)} papers a Neo4j...")
        
        batch_size = config.batch_size_import
        
        with self.driver.session(database=config.neo4j_database) as session:
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i + batch_size]
                
                with session.begin_transaction() as tx:
                    for paper in batch:
                        try:
                            self._import_paper(tx, paper)
                        except Exception as e:
                            logger.error(f"Error importando paper {paper.get('openalex_id', 'unknown')}: {e}")
                            continue
                
                logger.info(f"✓ Importado batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
        
        logger.info("✅ Importación completada")
    
    def _import_paper(self, tx, paper: Dict):
        """Importar un paper individual con todas sus relaciones"""
        openalex_id = paper["openalex_id"]
        
        # 1. Crear Paper
        tx.run("""
            MERGE (p:Paper {openalex_id: $openalex_id})
            SET p.doi = $doi,
                p.title = $title,
                p.abstract = $abstract,
                p.publication_year = $publication_year,
                p.publication_date = $publication_date,
                p.type = $type,
                p.language = $language,
                p.cited_by_count = $cited_by_count,
                p.author_count = $author_count,
                p.concept_count = $concept_count,
                p.reference_count = $reference_count,
                p.updated_at = $updated_at
        """, 
        openalex_id=openalex_id,
        doi=paper["doi"],
        title=paper["title"],
        abstract=paper["abstract"],
        publication_year=paper["publication_year"],
        publication_date=paper["publication_date"],
        type=paper["type"],
        language=paper["language"],
        cited_by_count=paper["cited_by_count"],
        author_count=len(paper["authorships"]),
        concept_count=len(paper["concepts"]),
        reference_count=len(paper["referenced_works"]),
        updated_at=paper["updated_at"]
        )
        
        # 2. Crear Journal si existe
        location_info = paper["location_info"]
        if location_info.get("journal_openalex_id"):
            tx.run("""
                MERGE (j:Journal {openalex_id: $journal_id})
                SET j.display_name = $journal_name,
                    j.issn_l = $issn_l,
                    j.type = $journal_type,
                    j.publisher = $publisher
                WITH j
                MATCH (p:Paper {openalex_id: $openalex_id})
                MERGE (p)-[:PUBLISHED_IN]->(j)
            """,
            journal_id=location_info["journal_openalex_id"],
            journal_name=location_info["journal_name"],
            issn_l=location_info["journal_issn_l"],
            journal_type=location_info["journal_type"],
            publisher=location_info["publisher"],
            openalex_id=openalex_id
            )
        
        # 3. Crear Authors e Institutions
        for authorship in paper["authorships"]:
            author_id = authorship["openalex_id"]
            if not author_id:
                continue
                
            # Crear autor
            tx.run("""
                MERGE (a:Author {openalex_id: $author_id})
                SET a.display_name = $display_name,
                    a.orcid = $orcid
                WITH a
                MATCH (p:Paper {openalex_id: $openalex_id})
                MERGE (a)-[r:AUTHORED]->(p)
                SET r.author_position = $position,
                    r.raw_author_name = $raw_name
            """,
            author_id=author_id,
            display_name=authorship["display_name"],
            orcid=authorship["orcid"],
            openalex_id=openalex_id,
            position=authorship["author_position"],
            raw_name=authorship["raw_author_name"]
            )
            
            # Crear instituciones del autor
            for institution in authorship["institutions"]:
                inst_id = institution["openalex_id"]
                if not inst_id:
                    continue
                    
                tx.run("""
                    MERGE (i:Institution {openalex_id: $inst_id})
                    SET i.display_name = $display_name,
                        i.country_code = $country_code,
                        i.type = $type,
                        i.ror = $ror
                    WITH i
                    MATCH (a:Author {openalex_id: $author_id})
                    MERGE (a)-[:AFFILIATED_WITH]->(i)
                    WITH i
                    MATCH (p:Paper {openalex_id: $openalex_id})
                    MERGE (p)-[:HAS_AFFILIATION]->(i)
                """,
                inst_id=inst_id,
                display_name=institution["display_name"],
                country_code=institution["country_code"],
                type=institution["type"],
                ror=institution["ror"],
                author_id=author_id,
                openalex_id=openalex_id
                )
                
                # Crear país si existe
                if institution["country_code"]:
                    tx.run("""
                        MERGE (c:Country {code: $country_code})
                        WITH c
                        MATCH (i:Institution {openalex_id: $inst_id})
                        MERGE (i)-[:LOCATED_IN]->(c)
                    """,
                    country_code=institution["country_code"],
                    inst_id=inst_id
                    )
        
        # 4. Crear Concepts
        for concept in paper["concepts"]:
            concept_id = concept["openalex_id"]
            if not concept_id:
                continue
                
            tx.run("""
                MERGE (c:Concept {openalex_id: $concept_id})
                SET c.display_name = $display_name,
                    c.level = $level,
                    c.wikidata = $wikidata
                WITH c
                MATCH (p:Paper {openalex_id: $openalex_id})
                MERGE (p)-[r:HAS_CONCEPT]->(c)
                SET r.score = $score
            """,
            concept_id=concept_id,
            display_name=concept["display_name"],
            level=concept["level"],
            wikidata=concept["wikidata"],
            openalex_id=openalex_id,
            score=concept["score"]
            )
        
        # 5. Crear Referencias
        for ref_id in paper["referenced_works"]:
            if ref_id:
                tx.run("""
                    MERGE (ref:Paper {openalex_id: $ref_id})
                    WITH ref
                    MATCH (p:Paper {openalex_id: $openalex_id})
                    MERGE (p)-[:CITES]->(ref)
                """,
                ref_id=ref_id,
                openalex_id=openalex_id
                )
    
    def get_database_stats(self):
        """Obtener estadísticas de la base de datos"""
        with self.driver.session(database=config.neo4j_database) as session:
            stats = {}
            
            # Contar nodos
            node_types = ["Paper", "Author", "Journal", "Institution", "Concept", "Country"]
            for node_type in node_types:
                count = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count").single()["count"]
                stats[f"{node_type}s"] = count
            
            # Contar relaciones
            rel_types = ["AUTHORED", "PUBLISHED_IN", "HAS_CONCEPT", "CITES", "AFFILIATED_WITH", "LOCATED_IN"]
            for rel_type in rel_types:
                count = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count").single()["count"]
                stats[rel_type] = count
            
            # Papers con abstracts
            with_abstract = session.run("""
                MATCH (p:Paper) 
                WHERE p.abstract IS NOT NULL AND p.abstract <> ''
                RETURN count(p) as count
            """).single()["count"]
            stats["Papers_with_abstract"] = with_abstract
            
        return stats
    
    def run_import(self, query: str, clear_db: bool = False):
        """Ejecutar importación completa con OpenAlex como fuente primaria"""
        logger.info("\n" + "="*80)
        logger.info("🚀 OPENALEX PRIMARY IMPORTER")
        logger.info("="*80)
        
        # Conectar a Neo4j
        if not self.connect_to_neo4j():
            return False
        
        try:
            # Limpiar DB si se requiere
            if clear_db:
                self.clear_database()
            
            # Crear constraints
            self.create_constraints_and_indexes()
            
            # Buscar, procesar e importar en batches simultáneos
            total_imported = self.search_and_process_openalex(query)
            
            if total_imported == 0:
                logger.error("No se importaron papers")
                return False
            
            # Mostrar estadísticas finales
            stats = self.get_database_stats()
            logger.info("\n📊 ESTADÍSTICAS FINALES:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:,}")
            
            logger.info(f"\n✅ IMPORTACIÓN COMPLETADA - {total_imported} papers importados!")
            return True
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Proceso interrumpido por el usuario")
            stats = self.get_database_stats()
            logger.info("📊 Estado actual de la base de datos:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:,}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error durante la importación: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        finally:
            if self.driver:
                self.driver.close()

def main():
    """Función principal para ejecución interactiva"""
    importer = OpenAlexImporter()
    
    print("\n🔬 OPENALEX PRIMARY IMPORTER")
    print("="*50)
    
    query = input("👉 Ingresa tu consulta de búsqueda: ").strip()
    if not query:
        print("❌ Query vacía. Saliendo...")
        return
    
    clear_db = input("🗑️  ¿Limpiar base de datos? (s/n): ").lower().startswith('s')
    
    print(f"\n📋 Configuración:")
    print(f"  • Query: '{query}'")
    print(f"  • Max papers: {config.max_papers_import}")
    print(f"  • Batch size: {config.batch_size_import}")
    print(f"  • Database: {config.neo4j_database}")
    print(f"  • Clear DB: {clear_db}")
    
    confirm = input("\n✅ ¿Continuar? (s/n): ").lower().startswith('s')
    if not confirm:
        print("❌ Cancelado por el usuario")
        return
    
    # Ejecutar importación
    success = importer.run_import(query, clear_db)
    
    if success:
        print("\n🎉 ¡Importación completada exitosamente!")
        print("\n💡 Consultas de ejemplo para explorar los datos:")
        print("  • MATCH (p:Paper) RETURN count(p) as total_papers")
        print("  • MATCH (a:Author)-[:AUTHORED]->(p:Paper) RETURN a.display_name, count(p) as papers ORDER BY papers DESC LIMIT 10")
        print("  • MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept) RETURN c.display_name, count(p) as papers ORDER BY papers DESC LIMIT 10")
        print("  • MATCH (p:Paper) WHERE p.abstract IS NOT NULL RETURN count(p) as papers_with_abstract")
    else:
        print("\n❌ Error durante la importación. Revisa los logs para más detalles.")

if __name__ == "__main__":
    main()