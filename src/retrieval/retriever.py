"""
Retrieval Module with Multiple Search Strategies
Implements dense (vector), sparse (BM25), and MMR-based retrieval with rank fusion.
"""

from typing import List, Literal
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


class UPBRetriever:
    """
    Multi-strategy retriever for UPB career documents.
    Supports: similarity search, MMR, BM25, and hybrid retrieval.
    """
    
    def __init__(self, chunks: List[Document], vectorstore=None):
        """
        Initialize retriever with document chunks.
        
        Args:
            chunks: List of chunked Document objects
            vectorstore: Optional FAISS/ChromaDB vectorstore for dense retrieval
        """
        self.chunks = chunks
        self.vectorstore = vectorstore
        self._bm25_retriever = None
        
    def get_bm25_retriever(self, k: int = 4) -> BM25Retriever:
        """
        Get or create BM25 retriever for sparse keyword-based search.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            BM25Retriever instance
        """
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever.from_documents(self.chunks)
        
        self._bm25_retriever.k = k
        return self._bm25_retriever
    
    def get_dense_retriever(self, k: int = 4, search_type: Literal["similarity", "mmr"] = "similarity"):
        """
        Get dense retriever from vectorstore.
        
        Args:
            k: Number of documents to retrieve
            search_type: "similarity" for standard search, "mmr" for diverse results
            
        Returns:
            Vectorstore retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please create embeddings first.")
        
        if search_type == "mmr":
            # MMR for diversity - reduces redundancy in results
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 5,  # Fetch more candidates for diversity
                    "lambda_mult": 0.7  # Balance: 1.0=relevance, 0.0=diversity
                }
            )
        else:
            # Standard similarity search
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def get_hybrid_retriever(self, k: int = 4, weights: List[float] = None):
        """
        Get hybrid retriever combining BM25 (sparse) and vector (dense) search.
        Uses rank fusion to intelligently merge results from both approaches.
        
        Args:
            k: Number of documents to retrieve
            weights: [bm25_weight, vector_weight]. Default: [0.5, 0.5]
            
        Returns:
            EnsembleRetriever with rank fusion combining both approaches
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please create embeddings first.")
        
        weights = weights or [0.5, 0.5]
        
        bm25_retriever = self.get_bm25_retriever(k=k)
        dense_retriever = self.get_dense_retriever(k=k)
        
        # EnsembleRetriever uses Reciprocal Rank Fusion (RRF) algorithm
        # which is more sophisticated than simple concatenation
        return EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=weights
        )
    
    def retrieve(
        self,
        query: str,
        method: Literal["bm25", "similarity", "mmr", "hybrid"] = "hybrid",
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents using specified method.
        
        Args:
            query: Search query
            method: Retrieval strategy
                - "bm25": Sparse keyword-based (no embeddings needed)
                - "similarity": Dense vector similarity search
                - "mmr": Maximal Marginal Relevance (diverse results)
                - "hybrid": Combination of BM25 + vector search
            k: Number of documents to retrieve
            **kwargs: Additional arguments for specific retrievers
            
        Returns:
            List of relevant Document objects
        """
        if method == "bm25":
            retriever = self.get_bm25_retriever(k=k)
        elif method == "similarity":
            retriever = self.get_dense_retriever(k=k, search_type="similarity")
        elif method == "mmr":
            retriever = self.get_dense_retriever(k=k, search_type="mmr")
        elif method == "hybrid":
            weights = kwargs.get("weights", [0.5, 0.5])
            retriever = self.get_hybrid_retriever(k=k, weights=weights)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        return retriever.invoke(query)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from loader.ingest import load_upb_documents
    from processing.chunking import chunk_documents
    
    print("Loading and chunking documents...\n")
    documents = load_upb_documents()
    chunks = chunk_documents(documents)
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Initialize retriever (without vectorstore for BM25 demo)
    retriever = UPBRetriever(chunks)
    
    # Test BM25 retrieval
    print("=" * 70)
    print("TESTING BM25 RETRIEVAL (keyword-based)")
    print("=" * 70)
    query = "ingeniería de sistemas inteligencia artificial"
    results = retriever.retrieve(query, method="bm25", k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} documents\n")
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Category: {doc.metadata.get('category', 'N/A')}")
        print(f"  Preview: {doc.page_content[:150]}...")
        print()
    
    print(" Retrieval module ready!")
    print("\nNote: For similarity, MMR, and hybrid search, initialize with a vectorstore.")
