# UPB RAG Career Exploration Assistant

A Retrieval-Augmented Generation (RAG) system to help prospective students explore UPB's engineering programs conversationally. The system ingests official UPB degree pages and uses Azure OpenAI to answer questions about programs, curricula, and career profiles.

## Features

- **Smart Data Collection**: Scrapes UPB engineering program pages using Playwright (handles dynamic JavaScript content)
- **Conversational Interface**: Ask free-form questions like "Which degree fits if I like AI and math?"
- **RAG-Powered Answers**: Retrieves relevant information and generates contextual responses using GPT-4.1 nano
- **Modular Architecture**: Easy to switch between vector databases (currently FAISS, ready for ChromaDB)

## Tech Stack

- **LLM**: Azure ChatOpenAI (GPT-4.1 nano)
- **Embeddings**: Azure OpenAI Embeddings (text-embedding-3-small)
- **Vector Database**: FAISS (modular design for easy switching)
- **Framework**: LangChain
- **Web Scraping**: Playwright + BeautifulSoup4

## Project Structure

```
.
├── src/
│   ├── config.py           # URLs configuration for all engineering programs
│   ├── data_loader.py      # Custom UPB page scraper using Playwright
│   ├── document_utils.py   # Utilities to load processed documents
│   ├── ingest.py          # Main data ingestion pipeline
│   └── rag.py             # RAG implementation (WIP)
├── data/
│   ├── raw_html/          # Saved HTML pages for debugging
│   └── processed/         # JSON files with extracted content
└── pyproject.toml         # Dependencies managed with uv

```

## 🚀 Getting Started

### 1. Data Ingestion

The first step is scraping UPB engineering program pages:

```bash
# Test mode: Scrape 2 programs (Systems Engineering & Data Science)
python3 src/ingest.py

# Full mode: Scrape all engineering programs
python3 src/ingest.py --full
```

**What it does:**
- Fetches fully rendered HTML using Playwright (handles JavaScript-heavy pages)
- Extracts relevant content (titles, paragraphs, sections)
- Saves raw HTML to `data/raw_html/` for debugging
- Saves processed documents to `data/processed/upb_careers_test.json` (or `upb_careers_all.json`)

### 2. Load Documents for RAG

Use the utility module to easily load documents:

```python
from document_utils import load_test_documents, print_document_stats

# Load documents
docs = load_test_documents()

# Show statistics
print_document_stats(docs)
```

## 📋 Available Engineering Programs

The system is configured to scrape the following UPB Medellín engineering programs:

- Ingeniería de Sistemas e Informática
- Ingeniería en Ciencia de Datos
- Ingeniería Administrativa
- Ingeniería Electrónica
- Ingeniería Eléctrica
- Ingeniería Mecánica
- Ingeniería Industrial
- Ingeniería Ambiental
- Ingeniería Química
- Ingeniería Aeronáutica
- Ingeniería en Diseño de Entretenimiento Digital
- Ingeniería en Nanotecnología

## Configuration

Create a `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4.1-nano
```
