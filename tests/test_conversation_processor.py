"""
Script de prueba para ConversationProcessor.
Ejecuta: python test_conversation_processor.py
"""

import asyncio
import sys
import os

# Agregar src al path para importar euler_mcp (desde tests/ hacia src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from euler_mcp.conversation_processor import ConversationProcessor


async def test_processor():
    """Prueba básica del ConversationProcessor."""
    
    print("🧠 Inicializando ConversationProcessor...")
    processor = ConversationProcessor()
    
    # Texto de prueba simulando una conversación de aprendizaje
    sample_conversation = """
    User: I'm learning TypeScript and I'm confused about interfaces vs types. 
    Can you explain the difference?
    
    Assistant: Great question! In TypeScript, both interfaces and types can define 
    object shapes, but there are key differences. Interfaces are more extensible 
    and better for object-oriented programming, while types are more flexible 
    for union types and complex type manipulations.
    
    User: How do generics work in TypeScript? I want to understand them better.
    
    Assistant: Generics in TypeScript allow you to write reusable code that works 
    with multiple types. They're like variables but for types instead of values.
    """
    
    print("📝 Analizando conversación de prueba...")
    
    # Extraer temas
    topics = await processor.extract_topics(sample_conversation)
    
    print(f"\n🎯 Temas encontrados ({len(topics)}):")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic['name']}")
        print(f"     Categoría: {topic['category']}")
        print(f"     Confianza: {topic['confidence']}")
        print(f"     Profundidad: {topic['depth']}")
        print(f"     Frecuencia: {topic['frequency']}")
        print()
    
    # Identificar relaciones
    relationships = await processor.identify_relationships(topics)
    
    print(f"🕸️ Relaciones encontradas ({len(relationships)}):")
    for i, rel in enumerate(relationships, 1):
        print(f"  {i}. {rel['source']} → {rel['target']}")
        print(f"     Tipo: {rel['type']}")
        print(f"     Peso: {rel['weight']}")
        print()
    
    print("✅ Prueba completada exitosamente!")


if __name__ == "__main__":
    # Ejecutar prueba asíncrona
    asyncio.run(test_processor())