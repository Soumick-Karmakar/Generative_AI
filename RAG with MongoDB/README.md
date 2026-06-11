# RAG with MongoDB - Retrieval Augmented Generation Pipeline

## 📋 Overview

This project implements a **Retrieval Augmented Generation (RAG)** pipeline using **MongoDB Atlas Vector Search** and **Google Gemini API**. The pipeline demonstrates a complete end-to-end workflow for ingesting documents, creating vector embeddings, performing semantic searches, and generating AI-powered responses using retrieved context.

The RAG pattern enhances LLM responses by:
1. Retrieving relevant documents from a vector database
2. Using those documents as context for the LLM
3. Generating more accurate, contextually-aware responses

---

## 🏗️ Architecture

The pipeline consists of three main phases:

### Phase 1: Data Ingestion
- Load documents (PDFs, text files, etc.)
- Chunk documents into manageable sizes
- Generate vector embeddings using Google Gemini
- Store documents and embeddings in MongoDB

### Phase 2: Data Retrieval
- Create vector search indexes in MongoDB
- Execute vector similarity searches
- Retrieve top-K most relevant documents
- Return context for LLM generation

### Phase 3: Generation
- Construct prompts with retrieved context
- Call Google Gemini for response generation
- Produce contextually accurate answers

```
Documents → Chunking → Embeddings → MongoDB Storage
                                         ↓
                                    Vector Search
                                         ↓
                    Retrieved Docs + User Query → LLM → Response
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- MongoDB Atlas account with Vector Search capability
- Google Cloud account with Gemini API access
- pip or uv package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "RAG with MongoDB"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   # or
   source .venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or using the pyproject.toml:
   ```bash
   pip install -e .
   ```

### Configuration

Create a `.env` file in the project root directory with the following variables:

```env
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# MongoDB Atlas Configuration
MONGO_USERNAME=your_mongodb_username
MONGO_PASSWORD=your_mongodb_password
MONGO_CLUSTER_URL=your_cluster_url.mongodb.net
```

**How to obtain these credentials:**

- **Google API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create an API key for Gemini
- **MongoDB Credentials**: 
  1. Create a cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
  2. Create a database user in Security → Database Access
  3. Get your cluster URL from the connection string

---

## 📁 Project Structure

```
RAG with MongoDB/
├── main.py                 # Entry point (boilerplate)
├── rag_mongo.py           # Main RAG pipeline implementation
├── test.py                # Test scripts for API connectivity
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
└── .venv/               # Virtual environment (auto-generated)
```

### File Descriptions

- **rag_mongo.py**: Contains the complete RAG pipeline including:
  - `get_embedding()`: Generates vector embeddings using Google Gemini
  - `get_query_results()`: Performs vector search queries in MongoDB
  - Document loading, chunking, and storage logic
  - LLM integration for response generation

- **test.py**: Utility scripts to test Google API connectivity and embedding/generation functionality

- **requirements.txt**: Lists all Python package dependencies

---

## 🔧 Usage

### Running the Complete Pipeline

Execute the main RAG pipeline:

```bash
python rag_mongo.py
```

This will:
1. Load a PDF document from MongoDB's investor relations
2. Split it into chunks
3. Generate embeddings for each chunk
4. Store documents in MongoDB Atlas
5. Create a vector search index
6. Execute a sample query
7. Retrieve relevant documents
8. Generate a response using Gemini

### Testing API Connectivity

Test your Google Gemini API setup:

```bash
python test.py
```

---

## 🔑 Key Components

### 1. Embedding Generation

```python
def get_embedding(text, input_type="document"):
    task_type = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values
```

- Uses Google's `gemini-embedding-001` model
- Generates 768-dimensional vectors
- Handles both document and query embeddings appropriately

### 2. Vector Search Query

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embeddings,
            "numCandidates": 100,
            "limit": 5
        }
    }
]
```

- Performs cosine similarity search
- Retrieves top-5 most relevant documents
- Uses MongoDB's native vector search aggregation

### 3. LLM-Based Generation

```python
completion = gemini_client.models.generate_content(
    model=GENERATION_MODEL,
    contents=prompt,
)
```

- Uses `gemini-2.5-flash` for fast response generation
- Includes retrieved context in the prompt
- Enables context-aware, accurate responses

---

## 📊 Technologies Used

| Technology | Purpose | Version |
|-----------|---------|---------|
| **MongoDB Atlas** | Vector database and storage | Latest |
| **Google Gemini API** | Embeddings & LLM generation | Latest |
| **LangChain** | Document loading & text splitting | 1.3.4+ |
| **PyMongo** | MongoDB Python driver | 4.17.0+ |
| **PyPDF** | PDF document loading | 6.13.1+ |
| **Python** | Programming language | 3.12+ |

---

## ⚙️ Configuration Details

### MongoDB Vector Search Index

The pipeline creates a vector search index with these specifications:

- **Field**: `embedding`
- **Type**: Vector search
- **Dimensions**: 768 (from Google Gemini embeddings)
- **Similarity**: Cosine
- **Candidates**: 100
- **Result Limit**: 5

### Embedding Model Details

- **Model**: `gemini-embedding-001`
- **Dimensions**: 768
- **Task Types**:
  - `RETRIEVAL_DOCUMENT`: For document embeddings
  - `RETRIEVAL_QUERY`: For query embeddings

### Generation Model Details

- **Model**: `gemini-2.5-flash`
- **Purpose**: Fast, efficient response generation
- **Features**: Supports prompt engineering with context

---

## 🔍 Understanding the Pipeline Flow

### Step 1: Data Ingestion
```
PDF/Document → Load with PyPDFLoader
            → Chunk with RecursiveCharacterTextSplitter (400 chars, 20 overlap)
            → Generate embeddings for each chunk
            → Store in MongoDB collection
```

### Step 2: Vector Search Index Creation
```
MongoDB Collection → Create SearchIndexModel
                   → Define vector field properties
                   → Index type: vectorSearch
                   → Build and activate index
```

### Step 3: Query & Retrieval
```
User Query → Generate query embedding
           → Execute vector search aggregation
           → Retrieve top-5 similar documents
           → Format context string
```

### Step 4: Response Generation
```
Context + Query → Create prompt with context
                → Call Gemini generate_content
                → Return AI-generated response
```

---

## 💡 Advanced Usage

### Using with Different Document Sources

Replace the PDF URL in `rag_mongo.py`:

```python
# Load from local file
loader = PyPDFLoader("path/to/local/document.pdf")

# The rest remains the same
data = loader.load()
```

### Customizing Chunk Size

Modify the text splitter parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Increase for longer contexts
    chunk_overlap=50     # Increase for more overlap
)
```

### Adjusting Search Parameters

Modify the vector search pipeline:

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": vector_index,
            "path": "embedding",
            "queryVector": query_embeddings,
            "numCandidates": 200,  # More candidates
            "limit": 10            # Return more results
        }
    }
]
```

### Custom Query Examples

```python
# Query different topics
queries = [
    "What are MongoDB's financial results?",
    "How does MongoDB handle security?",
    "What is MongoDB's product roadmap?"
]

for query in queries:
    results = get_query_results(query)
    # Process results...
```

---

## 📝 Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | `AIzaSy...` |
| `MONGO_USERNAME` | MongoDB Atlas username | `your_username` |
| `MONGO_PASSWORD` | MongoDB Atlas password | `your_password` |
| `MONGO_CLUSTER_URL` | MongoDB cluster connection URL | `cluster0.abcde.mongodb.net` |

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'google'`
- **Solution**: Ensure `google-genai` is installed: `pip install google-genai>=2.8.0`

**Issue**: MongoDB connection fails
- **Solution**: 
  - Verify credentials in `.env` file
  - Check MongoDB Atlas IP whitelist includes your IP
  - Ensure cluster is running

**Issue**: Vector search index not found
- **Solution**: 
  - Uncomment the `collection.create_search_index()` line
  - Wait for index creation to complete (usually takes a few minutes)
  - Comment it out for subsequent runs

**Issue**: API rate limits exceeded
- **Solution**: 
  - Reduce number of documents processed
  - Add delays between API calls
  - Use batch processing

### Logging & Debugging

Enable detailed output by uncommenting print statements in `rag_mongo.py`:

```python
print("Number of retrieved docs:", len(context_docs))
print("Context length:", len(context_string))
print("Prompt length:", len(prompt))
print(context_docs[:2])
```

---

## 🎯 Alternative Vector Store Implementations

The current implementation uses MongoDB Atlas native vector search. However, in production systems, these abstractions are commonly used:

- **LangChain + MongoDB Atlas Vector Search** - Recommended for flexibility
- **LangChain + FAISS** - Local vector storage
- **LangChain + Chroma** - Lightweight vector DB
- **LangChain + Pinecone** - Managed vector search service
- **LangChain + Weaviate** - Graph-based vector DB

For production systems, consider using LangChain's MongoDBAtlasVectorSearch class for cleaner abstractions.

---

## 📚 Learning Resources

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/overview/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [RAG Architecture Guide](https://www.mongodb.com/developer/products/mongodb/rag-your-data-with-mongodb-atlas-vector-search-and-langchain/)

---

## 📄 Notes

- Ensure your MongoDB Atlas cluster supports Vector Search (M10 or higher)
- The vector index creation is a one-time operation; subsequent runs can skip this step
- Document chunking strategy affects retrieval quality - adjust based on your use case
- Experiment with different models and parameters for optimal results

---

## 📧 Support & Issues

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all environment variables are correctly set
3. Test individual components (embedding, search, generation)
4. Check API quotas and limits on Google Cloud and MongoDB Atlas

---

## 📝 License

This project is for educational and development purposes.

---

**Last Updated**: 2026-06-11
**Version**: 0.1.0
