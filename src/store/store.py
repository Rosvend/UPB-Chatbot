"""
Vector Store Module
Manages FAISS vector store for document embeddings.
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


class VectorStoreManager:
    """Manages FAISS vector store operations."""
    
    def __init__(self, embeddings: AzureOpenAIEmbeddings | OpenAIEmbeddings):
        """
        Initialize vector store manager.
        
        Args:
            embeddings: Embeddings instance (Azure or OpenAI)
        """
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
    
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
        
        Returns:
            FAISS vector store instance
        """
        print(f"Creating embeddings for {len(documents)} documents...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("Vector store created")
        return self.vectorstore
    
    def save(self, path: str):
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Create one first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load(self, path: str) -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing saved vector store
        
        Returns:
            FAISS vector store instance
        """
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from {path}")
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            raise ValueError("No vector store exists. Create one first.")
        
        self.vectorstore.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("No vector store exists. Create or load one first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("No vector store exists. Create or load one first.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None):
        """
        Convert vector store to retriever.
        
        Args:
            search_type: "similarity" or "mmr"
            search_kwargs: Additional search parameters (e.g., {"k": 4})
        
        Returns:
            Retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("No vector store exists. Create or load one first.")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from embeddings.embeddings import get_embeddings
    from pipeline import prepare_documents_for_rag
    
    print("Vector Store Test\n")
    print("=" * 70)
    
    # Get chunks
    print("\nStep 1: Loading and chunking documents...")
    chunks = prepare_documents_for_rag(show_progress=False)
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize embeddings
    print("\nStep 2: Initializing embeddings...")
    try:
        embeddings = get_embeddings("azure")
        print("Using Azure OpenAI embeddings")
    except ValueError:
        try:
            embeddings = get_embeddings("openai")
            print("Using OpenAI embeddings")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Create vector store
    print("\nStep 3: Creating vector store...")
    vs_manager = VectorStoreManager(embeddings)
    vectorstore = vs_manager.create_from_documents(chunks)
    
    # Test search
    print("\nStep 4: Testing similarity search...")
    query = "ingeniería de sistemas inteligencia artificial"
    results = vs_manager.similarity_search(query, k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} documents\n")
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Category: {doc.metadata.get('category', 'N/A')}")
        print(f"  Preview: {doc.page_content[:150]}...")
        print()
    
    # Save vector store
    print("\nStep 5: Saving vector store...")
    save_path = "vectorstore/faiss_index"
    vs_manager.save(save_path)
    
    print("\nVector store test complete")
