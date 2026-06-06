# RAG Tutorial

A hands-on Python tutorial project for learning Retrieval-Augmented Generation (RAG) with LangChain, vector databases, Typesense, ChromaDB, FAISS, Groq, and LangGraph.

The repository contains small, practical examples that show how to load documents, split them into chunks, create embeddings, store those embeddings in a vector store, retrieve relevant context, and generate answers with an LLM.

## What This Project Covers

- Basic RAG concepts and project setup
- Document loading from text files and PDFs
- Text chunking for retrieval
- Embedding generation with Sentence Transformers
- Local vector storage with ChromaDB and FAISS
- Search and retrieval with Typesense
- LLM answer generation with Groq
- Agentic RAG workflow orchestration with LangGraph

## Project Structure

```text
.
|-- agentic_rag/
|   `-- agentic.py          # Agentic RAG example using LangGraph, FAISS, and Groq
|-- data/
|   |-- pdf/                # PDF documents used in examples
|   `-- text_files/         # Sample text documents
|-- notebook/
|   `-- document.py         # Document ingestion, embedding, ChromaDB, and RAG pipeline
|-- books.jsonl             # Sample book dataset for Typesense
|-- main.py                 # Basic project entry point and setup notes
|-- requirements.txt        # Python dependencies
|-- pyproject.toml          # Project metadata and uv dependencies
|-- typesense_db.py         # Typesense search and LangChain RAG example
`-- README.md
```

## Requirements

- Python 3.12 or newer
- A Groq API key for LLM generation
- A Typesense API key if you want to run the Typesense examples
- Optional Hugging Face token for faster model downloads and higher rate limits

## Setup

Clone or open the project folder, then create a virtual environment.

### Option 1: Using uv

```bash
uv venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
uv add -r requirements.txt
```

### Option 2: Using pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
Typesense_API_KEY=your_typesense_api_key_here
```

`GROQ_MODEL` is optional. If it is not set, the agentic RAG example uses `llama-3.1-8b-instant`.

For Hugging Face downloads, you can also set:

```env
HF_TOKEN=your_hugging_face_token_here
```

## How to Run

### Basic Project Check

```bash
python main.py
```

This prints a simple confirmation message and keeps the introductory RAG setup notes in one place.

### Document Ingestion and ChromaDB RAG Pipeline

```bash
python notebook/document.py
```

This script demonstrates:

- Creating sample text documents
- Loading text and PDF files
- Splitting documents into chunks
- Generating embeddings with `all-MiniLM-L6-v2`
- Storing embeddings in ChromaDB
- Retrieving relevant chunks
- Generating an answer with Groq

### Typesense Search and RAG Example

```bash
python typesense_db.py
```

This script demonstrates:

- Creating a Typesense collection
- Importing records from `books.jsonl`
- Running search queries
- Creating a LangChain Typesense vector store
- Running similarity search over `test.txt`

Make sure `Typesense_API_KEY` is available in your `.env` file before running this script.

### Agentic RAG with LangGraph

```bash
python agentic_rag/agentic.py
```

This example builds a small agentic workflow:

1. Decide whether retrieval is needed.
2. Retrieve relevant documents from a FAISS vector store.
3. Generate an answer using Groq.
4. End the graph.

The script also prints a Mermaid graph representation of the LangGraph workflow.

## Main Concepts

### Retrieval-Augmented Generation

RAG improves LLM responses by retrieving relevant information from an external knowledge source before generating an answer. Instead of relying only on model training data, the application provides fresh or domain-specific context at query time.

### Document Chunking

Large documents are split into smaller chunks so the retriever can find focused, relevant passages. Good chunking improves both search quality and answer quality.

### Embeddings

Embeddings convert text into numerical vectors. Similar text produces similar vectors, which makes semantic search possible.

### Vector Stores

Vector stores such as ChromaDB, FAISS, and Typesense make it possible to search documents by meaning instead of exact keyword matching.

### Agentic RAG

Agentic RAG adds decision-making around the retrieval process. Instead of always retrieving documents, the system can decide whether retrieval is needed and then route the query through the right workflow.

## Notes

- The `.env` file is ignored by Git and should not be committed.
- The first run of embedding examples may take longer because models need to be downloaded.
- `notebook/document.py` may create or update a local ChromaDB vector store.
- `typesense_db.py` uses a configured Typesense Cloud endpoint. Update the host, port, and protocol if you want to use a local Typesense server.

## Future Improvements

- Add a Streamlit or FastAPI interface
- Add automated tests for ingestion and retrieval
- Add configuration files for model and vector store settings
- Improve duplicate handling in vector stores
- Add examples for OpenAI, Ollama, or other LLM providers

## Author

Created as a learning project for experimenting with RAG pipelines and agentic AI workflows.
