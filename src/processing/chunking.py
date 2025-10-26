"""
Document Chunking Module
Splits documents into smaller chunks optimized for embedding and retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk in characters (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)
    
    Returns:
        list: List of chunked Document objects with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # Track position in original document
        separators=[
            "\n\n",  # Paragraphs (preferred)
            "\n",    # Lines
            " ",     # Words
            ""       # Characters (fallback)
        ]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from loader.ingest import load_upb_documents
    
    print("🚀 Loading documents...\n")
    documents = load_upb_documents()
    
    print(f"✅ Loaded {len(documents)} documents")
    print(f"📊 Total characters: {sum(len(doc.page_content) for doc in documents):,}\n")
    
    print("✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    
    print(f"\n✅ Created {len(chunks)} chunks")
    print(f"📊 Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks):,} characters")

    # Show chunks by category
    chunk_categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get('category', 'unknown')
        chunk_categories[cat] = chunk_categories.get(cat, 0) + 1
    
    print("\n📦 Chunks by category:")
    for cat, count in sorted(chunk_categories.items()):
        print(f"  - {cat}: {count} chunks")
    
    print("\n✨ Chunks ready for embedding!")