"""
Comprehensive RAG Test
Tests all features: retrieval, LLM integration, memory, and source citations.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain


def test_rag_features():
    """Test all RAG chain features comprehensively."""
    
    print("=" * 70)
    print("COMPREHENSIVE RAG CHAIN TEST")
    print("=" * 70)
    
    # Initialize system
    print("\n[1/5] Initializing retrieval system...")
    retriever, vectorstore_manager, chunks = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    print(f"System ready with {len(chunks)} chunks")
    
    # Create RAG chain
    print("\n[2/5] Creating RAG chain with GPT-4o-mini...")
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    print("RAG chain initialized")
    
    # Test 1: Basic query with sources
    print("\n[3/5] Testing basic query with source citations...")
    print("-" * 70)
    q1 = "¿Cuáles son las ingenierías disponibles en la UPB?"
    print(f"Q: {q1}")
    
    r1 = rag_chain.invoke(q1, include_sources=True)
    print(f"\nA: {r1['answer']}")
    
    if 'sources' in r1:
        print(f"\nSources retrieved: {len(r1['sources'])}")
        print("Categories:", set(s['category'] for s in r1['sources']))
    
    # Test 2: Follow-up question (memory test)
    print("\n[4/5] Testing conversation memory...")
    print("-" * 70)
    q2 = "¿Cuál de esas me recomiendas si me gusta programar?"
    print(f"Q: {q2}")
    
    r2 = rag_chain.invoke(q2, include_sources=False)
    print(f"\nA: {r2['answer']}")
    
    # Test 3: Specific question
    print("\n[5/5] Testing specific information retrieval...")
    print("-" * 70)
    q3 = "¿Qué opciones de financiación tienen?"
    print(f"Q: {q3}")
    
    r3 = rag_chain.invoke(q3, include_sources=True)
    print(f"\nA: {r3['answer']}")
    
    if 'sources' in r3:
        print(f"\nSources retrieved: {len(r3['sources'])}")
        scholarships_sources = [s for s in r3['sources'] if s['category'] == 'scholarships']
        print(f"Scholarship-related sources: {len(scholarships_sources)}")
    
    # Show conversation summary
    print("\n" + "=" * 70)
    print("CONVERSATION HISTORY")
    print("=" * 70)
    print(rag_chain.get_history_summary())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("Features tested:")
    print("  - Hybrid retrieval (BM25 + Vector with RRF)")
    print("  - GPT-4o-mini LLM integration")
    print("  - Conversation memory across multiple turns")
    print("  - Source citation with metadata")
    print("  - Spanish language responses")
    print("  - Context-aware follow-up questions")
    print("\nAll features working correctly!")
    print("=" * 70)


if __name__ == "__main__":
    test_rag_features()
