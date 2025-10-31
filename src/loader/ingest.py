"""
Document Loader Module
Loads markdown files from the data/ directory with metadata enrichment.
"""

import json
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_metadata_json():
    """
    Load the metadata.json file containing program information.
    
    Returns:
        dict: Program metadata dictionary
    """
    current_dir = Path(__file__).resolve().parent
    metadata_path = current_dir.parent.parent / "data" / "metadata" / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_upb_documents(show_progress=True, include_metadata_doc=True):
    """
    Load all markdown files from data/ directory and subdirectories.
    
    Args:
        show_progress: Whether to show progress bar (default: True)
        include_metadata_doc: Include metadata.json as a document (default: True)
    
    Returns:
        list: List of LangChain Document objects with content and metadata
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent.parent / "data"
    
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=show_progress,
        use_multithreading=True
    )
    
    documents = loader.load()
    
    for doc in documents:
        source_path = Path(doc.metadata['source'])
        relative_path = source_path.relative_to(data_dir)
        
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
    
    if include_metadata_doc:
        metadata = load_metadata_json()
        if metadata:
            metadata_content = f"""# UPB Engineering Programs Metadata

## Available Programs
Total programs: {metadata['metadata']['total_programs']}

"""
            for program in metadata['programs']:
                metadata_content += f"- {program['name']} ({', '.join(program['keywords'])})\n"
            
            metadata_content += f"\n## Accreditations\n"
            metadata_content += f"ABET Accredited: {', '.join(metadata['metadata']['abet_accredited'])}\n"
            metadata_content += f"Alta Calidad Accredited: {', '.join(metadata['metadata']['alta_calidad_accredited'])}\n"
            
            metadata_doc = Document(
                page_content=metadata_content,
                metadata={
                    'source': str(data_dir / 'metadata' / 'metadata.json'),
                    'category': 'metadata',
                    'type': 'program_catalog'
                }
            )
            documents.append(metadata_doc)
    
    return documents


if __name__ == "__main__":
    print("Loading markdown files from data/ directory...\n")
    documents = load_upb_documents()

    print(f"\nLoaded {len(documents)} documents")
    print(f"Total characters: {sum(len(doc.page_content) for doc in documents):,}")

    categories = {}
    for doc in documents:
        cat = doc.metadata.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print("\nDocuments by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")
