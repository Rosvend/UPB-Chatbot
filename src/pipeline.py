"""
Main Data Pipeline
Orchestrates the complete flow: load → chunk → ready for retrieval.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from loader.ingest import load_upb_documents
from processing.chunking import chunk_documents


def prepare_documents_for_rag(chunk_size=1000, chunk_overlap=200, show_progress=True, use_headers=True, add_context_prefix=True):
    """
    Complete data preparation pipeline.
    
    Args:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks in characters
        show_progress: Show loading progress bar
        use_headers: Use header-based chunking (default: True)
        add_context_prefix: Add contextual prefix to prevent hallucinations (default: True)
    
    Returns:
        list: Chunked documents ready for embedding and retrieval
    """
    documents = load_upb_documents(show_progress=show_progress)
    
    chunks = chunk_documents(
        documents, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        use_headers=use_headers,
        add_context_prefix=add_context_prefix
    )
    
    return chunks


if __name__ == "__main__":
    print("=" * 70)
    print("UPB RAG DATA PIPELINE")
    print("=" * 70)
    print("\nPipeline: Load → Chunk → Ready for Retrieval\n")
    
    chunks = prepare_documents_for_rag()
    
    print(f"\nPipeline complete!")
    print(f" Generated {len(chunks)} chunks")
    print(f" Average size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Statistics
    categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print("\nDistribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        percentage = (count / len(chunks)) * 100
        print(f"  - {cat}: {count} chunks ({percentage:.1f}%)")

    print("\nReady for embedding and retrieval!")
