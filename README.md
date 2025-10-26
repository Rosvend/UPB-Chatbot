# UPB RAG Career Exploration Assistant

A Retrieval-Augmented Generation (RAG) system to help prospective students explore UPB's engineering programs conversationally in Spanish. The system uses manually curated markdown documents and provides multi-strategy retrieval for accurate, context-aware responses.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Azure ChatOpenAI / OpenAI GPT-4.1 |
| **Embeddings** | AzureOpenAIEmbeddings |
| **Vector Store** | FAISS or ChromaDB |
| **Framework** | LangChain |
| **Retrieval** | BM25 (rank-bm25) + MMR |
| **UI** | Gradio *(planned)* |
| **Deployment** | Hugging Face Spaces *(planned)* |
| **Package Manager** | UV |

## Project Structure

```
.
├── data/                      # Curated markdown content
│   ├── about_upb.md          # University information
│   ├── contact/              # Contact information
│   ├── engineerings/         # Engineering program details (12 programs)
│   ├── enroll/               # Enrollment information
│   └── scholarships/         # Financial aid & scholarships
│
├── src/
│   ├── loader/
│   │   └── ingest.py        # Document loader with metadata enrichment
│   ├── processing/
│   │   └── chunking.py      # Smart text chunking module
│   ├── retrieval/
│   │   └── retriever.py     # Multi-strategy retriever (BM25/MMR/Hybrid)
│   └── pipeline.py          # Main orchestration pipeline
│
└── pyproject.toml           # Dependencies (UV)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/) package manager

### Configuration

Create a `.env` file with your Azure OpenAI credentials (for embeddings & LLM):

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini
```


### Installation

```bash
# Clone the repository
git clone https://github.com/Rosvend/UPB-RAG-Careers.git
cd UPB-RAG-Careers

# Install dependencies with UV
uv sync
```

### Running the Pipeline

#### 1. **Load Documents**
```bash
# Load all markdown files from data/
uv run python src/loader/ingest.py
```
**Output**: 16 documents loaded with category metadata

#### 2. **Chunk Documents**
```bash
# Split documents into optimized chunks
uv run python src/processing/chunking.py
```
**Output**: ~217 chunks (avg 792 chars)

#### 3. **Complete Pipeline**
```bash
# Run full pipeline: Load → Chunk
uv run python src/pipeline.py
```

#### 4. **Test Retrieval (BM25)**
```bash
# Test keyword-based retrieval
uv run python src/retrieval/retriever.py
```

## Module Documentation

### `src/loader/ingest.py`
Loads markdown files with automatic category detection based on folder structure.
- Supports progress tracking
- Multithreaded loading
- Metadata enrichment

### `src/processing/chunking.py`
Intelligent text splitting with context preservation.
- Paragraph-aware chunking
- Configurable size and overlap
- Preserves all metadata
- Tracks chunk position

### `src/retrieval/retriever.py`
Multi-strategy retrieval system.
- **BM25**: Keyword-based (Okapi BM25)
- **Similarity**: Dense vector search
- **MMR**: Diverse results with `lambda_mult` control
- **Hybrid**: Weighted ensemble of BM25 + vector

### `src/pipeline.py`
Orchestrates the complete data preparation flow.
- One-function interface
- Flexible configuration
- Detailed statistics output

