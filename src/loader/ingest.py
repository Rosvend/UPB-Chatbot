"""
UPB Career Data Ingestion Pipeline
Scrapes UPB engineering program pages and saves documents for RAG
"""

from pathlib import Path
import json
from data_loader import load_upb_careers
from config import UPB_ENGINEERING_URLS, TEST_URLS

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_documents_json(documents, output_file: Path):
    """Save documents to JSON file"""
    doc_dicts = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(documents)} documents to {output_file}")


def ingest_upb_data(test_mode: bool = False):
    """
    Main ingestion pipeline
    
    Args:
        test_mode: If True, only scrape TEST_URLS. Otherwise scrape all programs.
    """
    urls = TEST_URLS if test_mode else UPB_ENGINEERING_URLS
    
    print("=" * 70)
    print("UPB CAREER DATA INGESTION")
    print("=" * 70)
    print(f"Mode: {'TEST' if test_mode else 'FULL'}")
    print(f"URLs to scrape: {len(urls)}\n")
    
    # Load documents
    print("🚀 Starting data collection...\n")
    documents = load_upb_careers(urls, save_html=True)
    
    # Save processed documents
    output_file = PROCESSED_DIR / ("upb_careers_test.json" if test_mode else "upb_careers_all.json")
    save_documents_json(documents, output_file)
    
    # Print summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"✅ Documents loaded: {len(documents)}")
    print(f"📊 Total characters: {sum(doc.metadata['char_count'] for doc in documents):,}")
    print(f"📁 Raw HTML saved to: {RAW_HTML_DIR}")
    print(f"📁 Processed data saved to: {output_file}")
    
    # Show document titles
    print("\n📚 Loaded programs:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.metadata['title']} ({doc.metadata['char_count']:,} chars)")
    
    return documents


if __name__ == "__main__":
    import sys
    
    # Check if user wants full ingestion
    test_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_mode = False
        print("⚠️  Running FULL ingestion (all engineering programs)")
        print("This will take several minutes...\n")
    
    documents = ingest_upb_data(test_mode=test_mode)
    
    print("\n✨ Ingestion complete! Documents are ready for RAG processing.")