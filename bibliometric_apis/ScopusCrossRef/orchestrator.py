#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orquestador Principal - Neo4j Knowledge Graph Builder
Ejecuta los scripts en secuencia con procesamiento paralelo para scripts 3 y 4
"""

import os
import sys
import time
import logging
import hashlib
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_manager import get_config

# Configuración de logging principal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Mostrar banner inicial"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        NEO4J KNOWLEDGE GRAPH BUILDER                        ║
    ║                             ORCHESTRATOR v1.0                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Secuencia de ejecución:                                                     ║
    ║  1. Script 1: Reconstrucción de base de datos Neo4j                        ║
    ║  2. Script 2: Verificación y corrección de autores                         ║
    ║  3. Scripts 3 y 4 (en paralelo): Completar datos y enriquecimiento        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_search_query():
    """Solicitar query de búsqueda al usuario"""
    print("\n" + "="*80)
    print("CONFIGURACIÓN DE BÚSQUEDA")
    print("="*80)
    
    print("\nEjemplos de consultas de búsqueda:")
    print('1. TITLE-ABS-KEY(("plastic recycling" OR "recycled plastic") AND ("circular economy" OR "sustainability" OR "toxicity" OR "contamination"))')
    print('2. TITLE-ABS-KEY("artificial intelligence" AND "machine learning")')
    print('3. TITLE-ABS-KEY("renewable energy" AND "solar power")')
    
    while True:
        query = input("\nIngrese su consulta de búsqueda para Scopus: ").strip()
        
        if not query:
            print("❌ La consulta no puede estar vacía. Intente nuevamente.")
            continue
        
        # Validar formato básico
        if not any(keyword in query.upper() for keyword in ['TITLE-ABS-KEY', 'AUTHOR', 'AFFIL']):
            print("⚠️  Recomendación: Use formato Scopus como TITLE-ABS-KEY(\"términos\")")
            confirm = input("¿Desea continuar con esta consulta? (s/n): ")
            if confirm.lower() != 's':
                continue
        
        print(f"\n✅ Consulta configurada: {query}")
        return query

def get_execution_options():
    """Obtener opciones de ejecución del usuario"""
    print("\n" + "="*80)
    print("OPCIONES DE EJECUCIÓN")
    print("="*80)
    
    # Opción de limpiar base de datos
    clear_db = input("\n¿Desea limpiar completamente la base de datos Neo4j? (s/n): ").lower() == 's'
    
    # Opción de procesamiento paralelo
    parallel = True
    if hasattr(get_config(), 'enable_parallel_processing'):
        parallel = get_config().enable_parallel_processing
    
    if not parallel:
        enable_parallel = input("\n¿Desea habilitar procesamiento paralelo para scripts 3 y 4? (s/n): ").lower() == 's'
    else:
        enable_parallel = True
        print(f"\n✅ Procesamiento paralelo habilitado por configuración")
    
    # Confirmación final
    print(f"\n📋 RESUMEN DE CONFIGURACIÓN:")
    print(f"   - Limpiar base de datos: {'Sí' if clear_db else 'No'}")
    print(f"   - Procesamiento paralelo: {'Sí' if enable_parallel else 'No'}")
    
    confirm = input("\n¿Desea continuar con esta configuración? (s/n): ")
    if confirm.lower() != 's':
        print("❌ Ejecución cancelada por el usuario.")
        sys.exit(0)
    
    return {
        'clear_db': clear_db,
        'parallel': enable_parallel
    }

def execute_script(script_name, script_function, *args, **kwargs):
    """Ejecutar un script con manejo de errores y logging"""
    logger.info(f"🚀 Iniciando {script_name}...")
    start_time = time.time()
    
    try:
        result = script_function(*args, **kwargs)
        
        if result:
            execution_time = time.time() - start_time
            logger.info(f"✅ {script_name} completado exitosamente en {execution_time:.2f} segundos")
            return True
        else:
            logger.error(f"❌ {script_name} falló")
            return False
            
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"💥 Error crítico en {script_name} después de {execution_time:.2f} segundos: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def execute_parallel_scripts():
    """Ejecutar scripts 3 y 4 en paralelo"""
    logger.info("🔄 Iniciando ejecución paralela de Scripts 3 y 4...")
    
    # Importar las funciones de los scripts
    from script3_complete_data import run_script3
    from script4_crossref import run_script4
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Enviar ambos scripts para ejecución
        future_script3 = executor.submit(execute_script, "Script 3", run_script3)
        future_script4 = executor.submit(execute_script, "Script 4", run_script4)
        
        # Mapear futuros a nombres de scripts
        future_to_script = {
            future_script3: "Script 3",
            future_script4: "Script 4"
        }
        
        # Procesar completados
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            try:
                result = future.result()
                results[script_name] = result
                logger.info(f"🏁 {script_name} terminado con resultado: {result}")
            except Exception as e:
                logger.error(f"💥 Excepción en {script_name}: {e}")
                results[script_name] = False
    
    return results

def main():
    """Función principal del orquestador"""
    start_time = datetime.now()
    
    # Mostrar banner
    print_banner()
    
    try:
        # Cargar y validar configuración
        logger.info("📋 Validando configuración...")
        config = get_config()
        logger.info("✅ Configuración válida")
        
        # Obtener query de búsqueda
        search_query = get_search_query()
        
        # Generar hash de la query
        query_hash = hashlib.md5(search_query.encode()).hexdigest()[:8]
        logger.info(f"🔑 Hash de query generado: {query_hash}")
        
        # Obtener opciones de ejecución
        options = get_execution_options()
        
        logger.info(f"🎯 Iniciando orquestación con query: {search_query}")
        
        # Crear directorio de datos
        os.makedirs(config.data_dir, exist_ok=True)
        
        # PASO 1: Ejecutar Script 1 (Reconstrucción de base de datos)
        logger.info("\n" + "="*80)
        logger.info("PASO 1: RECONSTRUCCIÓN DE BASE DE DATOS")
        logger.info("="*80)
        
        from script1_neo4j_rebuild import run_script1
        
        script1_success = execute_script(
            "Script 1", 
            run_script1, 
            search_query, 
            options['clear_db'], 
            False  # interactive=False
        )
        
        if not script1_success:
            logger.error("💀 Script 1 falló. Deteniendo ejecución.")
            return False
        
        # PASO 2: Ejecutar Script 2 (Verificación de autores)
        logger.info("\n" + "="*80)
        logger.info("PASO 2: VERIFICACIÓN Y CORRECCIÓN DE AUTORES")
        logger.info("="*80)
        
        from script2_author_fix import run_script2
        
        script2_success = execute_script(
            "Script 2", 
            run_script2, 
            False,  # interactive=False
            query_hash  # ¡AGREGADO EL HASH!
        )
        
        if not script2_success:
            logger.error("💀 Script 2 falló. Deteniendo ejecución.")
            return False
        
        # PASO 3: Ejecutar Scripts 3 y 4 (en paralelo o secuencial)
        logger.info("\n" + "="*80)
        logger.info("PASO 3: COMPLETAR DATOS Y ENRIQUECIMIENTO")
        logger.info("="*80)
        
        if options['parallel']:
            # Ejecución paralela
            parallel_results = execute_parallel_scripts()
            
            script3_success = parallel_results.get("Script 3", False)
            script4_success = parallel_results.get("Script 4", False)
            
        else:
            # Ejecución secuencial
            from script3_complete_data import run_script3
            from script4_crossref import run_script4
            
            script3_success = execute_script("Script 3", run_script3)
            script4_success = execute_script("Script 4", run_script4)
        
        # RESUMEN FINAL
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("RESUMEN DE EJECUCIÓN")
        logger.info("="*80)
        
        results_summary = [
            ("Script 1 (Reconstrucción DB)", script1_success),
            ("Script 2 (Corrección Autores)", script2_success),
            ("Script 3 (Completar Datos)", script3_success),
            ("Script 4 (Enriquecimiento Crossref)", script4_success)
        ]
        
        for script_name, success in results_summary:
            status = "✅ ÉXITO" if success else "❌ FALLÓ"
            logger.info(f"{script_name}: {status}")
        
        all_success = all(success for _, success in results_summary)
        
        logger.info(f"\n⏱️  Tiempo total de ejecución: {total_time}")
        logger.info(f"🎯 Query utilizada: {search_query}")
        logger.info(f"🔑 Hash de query: {query_hash}")
        
        if all_success:
            logger.info("🎉 ¡ORQUESTACIÓN COMPLETADA EXITOSAMENTE!")
            print("\n🎉 ¡Todos los scripts se ejecutaron correctamente!")
            print("🔍 Revise los archivos de log individuales para más detalles:")
            print("   - neo4j_rebuild.log")
            print("   - author_fix.log")
            print("   - complete_data.log")
            print("   - crossref_enrichment.log")
            print("   - orchestrator.log")
        else:
            logger.warning("⚠️  ORQUESTACIÓN COMPLETADA CON ERRORES")
            print("\n⚠️  Algunos scripts fallaron. Revise los logs para más información.")
        
        return all_success
        
    except KeyboardInterrupt:
        logger.warning("🛑 Ejecución interrumpida por el usuario")
        print("\n🛑 Ejecución cancelada por el usuario.")
        return False
    except Exception as e:
        logger.error(f"💥 Error crítico en orquestador: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n💥 Error crítico: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)