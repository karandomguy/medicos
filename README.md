# medicos - Medical Information Retrieval System

## Overview

medicos is a Retrieval-Augmented Generation (RAG) system designed to provide accurate and well-referenced responses to medical queries. The system combines vector search capabilities, external search fallback, and large language model integration to deliver reliable medical information.

## Features

- **Vector Database Storage**: Uses ChromaDB to store and retrieve medical documents based on semantic similarity
- **Document Processing Pipeline**: Processes medical documents and websites, chunking content into manageable segments
- **Google Search Fallback**: Falls back to live Google search when local database lacks relevant information
- **Response Validation**: Validates database results for relevance before providing answers
- **Response Caching**: Caches responses to similar questions to improve performance
- **Medical-Specific Embeddings**: Uses domain-specific embedding models for better retrieval accuracy
- **Source Attribution**: Provides transparent source attribution for all information
- **RESTful API**: Exposes functionality through a FastAPI-based REST API

## System Architecture

The system is composed of three main components:

1. **Document Processor (`document_processing.py`)**: Handles document loading, chunking, and indexing
2. **RAG System (`rag_system.py`)**: Core component that manages retrieval, validation, generation, and caching
3. **API Server (`main.py`)**: FastAPI-based interface for external applications

## Prerequisites

- Python 3.8+
- Required API keys:
  - Hugging Face API key (for embedding models)
  - Google API key (for search fallback)
  - Google Custom Search Engine ID
  - Groq API key (for LLM integration)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/medicos.git
cd medicos
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following environment variables:
```
HUGGINGFACE_API_KEY=your_huggingface_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Process Medical Documents

The document processor can ingest documents from JSON files or URLs:

```python
from document_processing import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process medical websites
medical_urls = [
    "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
    "https://www.cdc.gov/diabetes/basics/diabetes.html"
]
processor.run_pipeline(urls=medical_urls)
```

### Query the System

```python
from rag_system import MedicalRAG

# Initialize RAG system
rag = MedicalRAG()

# Process a medical query
response = rag.process_medical_query(
    "What are the early symptoms of diabetes?",
    use_google_fallback=True,
    top_k=5
)

print(response["answer"])
```

### Running the API Server

```bash
python main.py
```

The API will be available at http://localhost:8000

### API Endpoints

- **POST /api/query**: Process a medical question
  - Request body: `{"question": "your question", "use_google_fallback": true, "top_k": 5}`
  - Returns: Answer with sources

- **GET /api/health**: Check system health
  - Returns: System status

## System Flow

1. **Query Processing**:
   - Check cache for similar questions
   - Search vector database (ChromaDB) for relevant documents
   - Validate relevance of retrieved documents
   - Fall back to Google Search if needed
   - Generate answer using LLM (Groq API)
   - Cache response for future similar queries

2. **Document Processing**:
   - Load documents from file or web
   - Chunk documents into manageable segments
   - Generate embeddings for each chunk
   - Store in vector database

## Customization

- **Embedding Model**: Change `embedding_model_name` in `MedicalRAG` initialization
- **LLM Model**: Change `llm_model` in `MedicalRAG` initialization
- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `DocumentProcessor`
- **Cache Expiry**: Adjust `cache_expiry_days` in `MedicalRAG`

## License

[Add your license information here]

## Contributors

[Add contributor information here]