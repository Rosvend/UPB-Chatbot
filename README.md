# UPB RAG Career Exploration Assistant

A Retrieval-Augmented Generation (RAG) system to help prospective students explore UPB's engineering programs conversationally in Spanish. The system uses manually curated markdown documents and provides multi-strategy retrieval for accurate, context-aware responses.

## Features

- Conversational AI with GPT-4o-mini for natural interactions
- Multi-strategy retrieval: BM25, Vector Similarity, MMR, and Hybrid RRF
- Conversation memory for multi-turn dialogues
- Source citations for transparency
- Spanish language optimized
- 16 curated documents covering 12 engineering programs
- 217 optimized chunks for efficient retrieval

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Azure ChatOpenAI / OpenAI GPT-4o-mini |
| **Embeddings** | AzureOpenAIEmbeddings (text-embedding-3-small) |
| **Vector Store** | FAISS (CPU version) |
| **Framework** | LangChain 1.0.2 |
| **Retrieval** | BM25 (rank-bm25) + MMR + RRF Ensemble |
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
│   ├── embeddings/
│   │   └── embeddings.py    # Azure/OpenAI embeddings initialization
│   ├── loader/
│   │   └── ingest.py        # Document loader with metadata enrichment
│   ├── processing/
│   │   └── chunking.py      # Smart text chunking module
│   ├── rag/
│   │   └── chain.py         # RAG chain with conversation memory
│   ├── retrieval/
│   │   └── retriever.py     # Multi-strategy retriever (BM25/MMR/Hybrid)
│   ├── vectorstore/
│   │   └── store.py         # FAISS vector store manager
│   ├── pipeline.py          # Document preparation pipeline
│   └── setup_retrieval.py   # Complete retrieval system setup
│
├── vectorstore/             # FAISS index files (gitignored)
└── pyproject.toml           # Dependencies (UV)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/) package manager


### Installation

```bash
# Clone the repository
git clone https://github.com/Rosvend/UPB-RAG-Careers.git
cd UPB-RAG-Careers

# Install dependencies with UV
uv sync
```

### Configuration

Create a `.env` file with your Azure OpenAI credentials (for embeddings & LLM):

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini
```

### Running the Pipeline

#### 1. **RAG Chain with Conversation** (Full System)
```bash
# Test complete RAG chain with GPT-4o-mini, memory, and source citations
uv run python src/rag/chain.py
```
**What it does**:
- Sets up complete retrieval system
- Initializes GPT-4o-mini LLM
- Tests multi-turn conversation
- Shows source citations
- Demonstrates conversation memory

**Output**: Complete conversational RAG system test

#### 2. **Complete Retrieval Setup**
```bash
# Set up embeddings, vector store, and all retrieval methods
uv run python src/setup_retrieval.py
```
**What it does**:
- Loads 16 markdown documents
- Creates ~217 optimized chunks
- Initializes Azure OpenAI embeddings
- Creates/loads FAISS vector store
- Tests all retrieval methods (BM25, Similarity, MMR, Hybrid)

**Output**: Fully initialized retrieval system ready for RAG

#### 3. **Individual Modules**

**Load Documents**
```bash
uv run python src/loader/ingest.py
```
**Output**: 16 documents with category metadata

**Chunk Documents**
```bash
uv run python src/processing/chunking.py
```
**Output**: ~217 chunks (avg 792 chars)

**Test Embeddings**
```bash
uv run python src/embeddings/embeddings.py
```
**Output**: Embedding model initialization test

**Test Vector Store**
```bash
uv run python src/vectorstore/store.py
```
**Output**: FAISS index creation, save, and load test

**Test Retrieval**
```bash
uv run python src/retrieval/retriever.py
```
**Output**: BM25 retrieval test (no embeddings needed)

## Module Documentation

### `src/embeddings/embeddings.py`
Manages embedding model initialization with dual provider support.
- **Azure OpenAI**: Primary provider with text-embedding-3-small
- **OpenAI**: Fallback provider
- Environment variable validation
- Test mode for verification

**Usage**:
```python
from embeddings.embeddings import get_embeddings

# Azure (default)
embeddings = get_embeddings(provider="azure")

# OpenAI fallback
embeddings = get_embeddings(provider="openai")
```

### `src/vectorstore/store.py`
FAISS vector store manager for efficient similarity search.
- Create index from documents
- Save/load to disk
- Incremental document additions
- Multiple search modes (similarity, MMR)
- Convert to retriever interface

**Key Features**:
- Persistent storage (saves to `vectorstore/faiss_index/`)
- Fast similarity search with FAISS CPU
- MMR support for diverse results
- Seamless integration with UPBRetriever

**Usage**:
```python
from vectorstore.store import VectorStoreManager
from embeddings.embeddings import get_embeddings

embeddings = get_embeddings()
manager = VectorStoreManager(embeddings)

# Create from documents
manager.create_from_documents(chunks)
manager.save("vectorstore/faiss_index")

# Load existing
manager.load("vectorstore/faiss_index")

# Search
results = manager.similarity_search("query", k=4)
```

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
Multi-strategy retrieval system with **Reciprocal Rank Fusion (RRF)**.
- **BM25**: Keyword-based sparse retrieval (Okapi BM25)
- **Similarity**: Dense vector search with embeddings
- **MMR**: Maximal Marginal Relevance for diverse results
- **Hybrid**: Ensemble with RRF algorithm (from `langchain-classic`)
  - Uses Reciprocal Rank Fusion to intelligently merge BM25 + vector results
  - Better than simple concatenation: boosts docs appearing in both retrievers
  - Handles different scoring scales and provides better diversity control

**Why RRF?** Documents that appear in both BM25 and vector search get higher scores,
indicating they're relevant both keyword-wise AND semantically. This produces better
results than either method alone.

**Usage**:
```python
from retrieval.retriever import UPBRetriever
from setup_retrieval import setup_retrieval_system

# Full setup
retriever, vectorstore_manager, chunks = setup_retrieval_system()

# Different retrieval strategies
query = "ingeniería de sistemas inteligencia artificial"

# BM25 only (keyword matching)
results = retriever.retrieve(query, method="bm25", k=4)

# Similarity search (semantic)
results = retriever.retrieve(query, method="similarity", k=4)

# MMR (diverse results)
results = retriever.retrieve(query, method="mmr", k=4)

# Hybrid with RRF (recommended)
results = retriever.retrieve(query, method="hybrid", k=4)

# Custom hybrid weights
results = retriever.retrieve(
    query, 
    method="hybrid", 
    k=4, 
    weights=[0.3, 0.7]  # [bm25_weight, vector_weight]
)
```

### `src/setup_retrieval.py`
Complete retrieval system initialization and testing.
- One-function setup for entire retrieval pipeline
- Automatic vector store creation/loading
- Multi-method comparison testing
- Production-ready configuration

**Quick Start**:
```python
from setup_retrieval import setup_retrieval_system

# Initialize everything
retriever, vectorstore_manager, chunks = setup_retrieval_system()

# Ready to use!
results = retriever.retrieve("your query", method="hybrid", k=4)
```

### `src/rag/chain.py`
Conversational RAG chain with GPT-4o-mini and memory.
- Multi-turn conversation support
- Conversation history tracking
- Source citations with document metadata
- Spanish language optimized prompts
- Hybrid retrieval integration

**Features**:
- Maintains context across multiple questions
- Provides document sources for transparency
- Friendly, professional tone in Spanish
- Suggests related programs when appropriate

**Usage**:
```python
from rag.chain import UPBRAGChain
from setup_retrieval import setup_retrieval_system

# Setup
retriever, _, _ = setup_retrieval_system()
rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")

# Ask questions
response = rag_chain.invoke(
    "¿Qué carrera debo estudiar si me gusta la IA?",
    include_sources=True
)

print(response['answer'])
for source in response['sources']:
    print(f"- {source['category']}: {source['source']}")

# Continue conversation (memory is maintained)
response2 = rag_chain.invoke("¿Qué requisitos necesito?")

# Clear history when needed
rag_chain.clear_history()
```

# Ready to use!
results = retriever.retrieve("your query", method="hybrid", k=4)
```

### `src/pipeline.py`
Orchestrates the complete data preparation flow.
- One-function interface
- Flexible configuration
- Detailed statistics output

## Quick Start Examples

### Interactive Chat
```bash
# Run interactive chat interface
uv run python src/example_usage.py
```

Type your questions in Spanish and the assistant will respond using the RAG system. Commands:
- `salir` - Exit the chat
- `limpiar` - Clear conversation history

### Programmatic Usage

**Basic RAG Query**:
```python
from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain

# Initialize
retriever, _, _ = setup_retrieval_system()
rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")

# Ask question
response = rag_chain.invoke("¿Qué es la ingeniería de sistemas?")
print(response['answer'])
```

**With Source Citations**:
```python
response = rag_chain.invoke(
    "¿Qué becas están disponibles?",
    include_sources=True
)

print(response['answer'])
print("\nFuentes:")
for source in response['sources']:
    print(f"- {source['category']}: {source['source']}")
```

**Multi-turn Conversation**:
```python
# First question
r1 = rag_chain.invoke("¿Qué ingenierías tienen?")

# Follow-up (uses conversation memory)
r2 = rag_chain.invoke("¿Cuál me recomiendas si me gusta programar?")

# Another follow-up
r3 = rag_chain.invoke("¿Cuánto dura ese programa?")

# Clear history when done
rag_chain.clear_history()
```

