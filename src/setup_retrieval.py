"""
Complete Retrieval Setup Script
Demonstrates how to set up the full retrieval pipeline with embeddings and vector store.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loader.ingest import load_upb_documents
from processing.chunking import chunk_documents
from embeddings.embeddings import get_embeddings
from store.store import VectorStoreManager
from retrieval.retriever import UPBRetriever


def setup_retrieval_system(
    vectorstore_path: str = "vectorstore/faiss_index",
    use_existing: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_headers: bool = True,
    add_context_prefix: bool = True,
):
    """
    Set up complete retrieval system with embeddings and vector store.
    
    Args:
        vectorstore_path: Path to save/load FAISS index
        use_existing: If True and vectorstore exists, load it. Otherwise create new.
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        use_headers: Use header-based chunking (default: True)
        add_context_prefix: Add contextual prefix to prevent hallucinations (default: True)
        
    Returns:
        Tuple of (UPBRetriever, VectorStoreManager, chunks)
    """
    print("=" * 70)
    print("UPB RAG - RETRIEVAL SYSTEM SETUP")
    print("=" * 70)
    
    print("\n[1/4] Loading documents...")
    documents = load_upb_documents(show_progress=True)
    print(f"Loaded {len(documents)} documents")
    
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(
        documents, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        use_headers=use_headers,
        add_context_prefix=add_context_prefix
    )
    print(f"Created {len(chunks)} chunks")
    
    print("\n[3/4] Initializing embeddings...")
    embeddings = get_embeddings(provider="azure")
    print("Embeddings ready")
    
    print("\n[4/4] Setting up vector store...")
    vectorstore_manager = VectorStoreManager(embeddings)
    
    if use_existing and Path(vectorstore_path).exists():
        print(f"Loading existing vector store from {vectorstore_path}...")
        vectorstore_manager.load(vectorstore_path)
        print("Vector store loaded")
    else:
        print("Creating new vector store...")
        vectorstore_manager.create_from_documents(chunks)
        print("Vector store created")
        
        print(f"Saving to {vectorstore_path}...")
        vectorstore_manager.save(vectorstore_path)
        print("Vector store saved")
    
    retriever = UPBRetriever(chunks, vectorstore=vectorstore_manager.vectorstore)
    
    print("\n" + "=" * 70)
    print("RETRIEVAL SYSTEM READY")
    print("=" * 70)
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding Model: Azure OpenAI")
    print(f"Vector Store: FAISS")
    print("\nAvailable retrieval methods:")
    print("  - bm25: Keyword-based sparse retrieval")
    print("  - similarity: Dense vector similarity search")
    print("  - mmr: Maximal Marginal Relevance (diverse results)")
    print("  - hybrid: BM25 + Vector search with RRF (recommended)")
    print("=" * 70)
    
    return retriever, vectorstore_manager, chunks


def test_all_retrieval_methods(retriever: UPBRetriever):
    """
    Test all retrieval methods with sample queries.
    
    Args:
        retriever: Initialized UPBRetriever instance
    """
    print("\n\n" + "=" * 70)
    print("TESTING ALL RETRIEVAL METHODS")
    print("=" * 70)
    
    test_queries = [
        "ingeniería de sistemas inteligencia artificial",
        "becas y financiación estudiantil",
        "requisitos de inscripción"
    ]
    
    methods = ["bm25", "similarity", "mmr", "hybrid"]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: '{query}'")
        print('=' * 70)
        
        for method in methods:
            print(f"\n--- {method.upper()} ---")
            try:
                results = retriever.retrieve(query, method=method, k=2)
                print(f"Retrieved {len(results)} documents:")
                for i, doc in enumerate(results, 1):
                    category = doc.metadata.get('category', 'N/A')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"  {i}. [{category}] {preview}...")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    # Setup the complete retrieval system
    retriever, vectorstore_manager, chunks = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True  # Use existing index if available
    )
    
    # Test all retrieval methods
    test_all_retrieval_methods(retriever)
    
    print("\n\n" + "=" * 70)
    print("QUICK START EXAMPLE")
    print("=" * 70)
    print("""
          
          
# To use the retrieval system in your code:

from setup_retrieval import setup_retrieval_system

# Initialize
retriever, vectorstore_manager, chunks = setup_retrieval_system()

# Use different retrieval methods
query = "ingeniería de sistemas"

# BM25 (keyword-based, no embeddings needed)
results = retriever.retrieve(query, method="bm25", k=4)

# Similarity search (dense vector)
results = retriever.retrieve(query, method="similarity", k=4)

# MMR for diverse results
results = retriever.retrieve(query, method="mmr", k=4)

# Hybrid (recommended - combines BM25 + vector with RRF)
results = retriever.retrieve(query, method="hybrid", k=4)

# Custom weights for hybrid
results = retriever.retrieve(
    query, 
    method="hybrid", 
    k=4, 
    weights=[0.3, 0.7]  # [bm25_weight, vector_weight]
)
""")
    print("=" * 70)
