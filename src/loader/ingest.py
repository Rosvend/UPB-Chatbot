"""
Document Loader Module
Loads markdown files from the data/ directory with metadata enrichment.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_upb_documents(show_progress=True):
    """
    Load all markdown files from data/ directory and subdirectories.
    
    Args:
        show_progress: Whether to show progress bar (default: True)
    
    Returns:
        list: List of LangChain Document objects with content and metadata
    """
    # Get data directory path
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent.parent / "data"
    
    # Load all .md files recursively
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=show_progress,
        use_multithreading=True
    )
    
    documents = loader.load()
    
    # Add source category to metadata based on subdirectory
    for doc in documents:
        source_path = Path(doc.metadata['source'])
        relative_path = source_path.relative_to(data_dir)
        
        # Determine category from subdirectory
        if relative_path.parts[0] == 'engineerings':
            doc.metadata['category'] = 'engineering'
        elif relative_path.parts[0] == 'contact':
            doc.metadata['category'] = 'contact'
        elif relative_path.parts[0] == 'enroll':
            doc.metadata['category'] = 'enrollment'
        elif relative_path.parts[0] == 'scholarships':
            doc.metadata['category'] = 'scholarships'
        else:
            doc.metadata['category'] = 'general'
    
    return documents


if __name__ == "__main__":
    print(" Loading markdown files from data/ directory...\n")
    documents = load_upb_documents()

    print(f"\n Loaded {len(documents)} documents")
    print(f" Total characters: {sum(len(doc.page_content) for doc in documents):,}")

    # Group by category
    categories = {}
    for doc in documents:
        cat = doc.metadata.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print("\n📚 Documents by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")