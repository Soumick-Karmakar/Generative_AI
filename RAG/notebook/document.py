#***********************************#
#                                   #
#      DATA INGESTION PIPELINE      #
#                                   #
#***********************************#

import os
from pathlib import Path

# Get the script's directory and set up relative paths
SCRIPT_DIR = Path(__file__).parent.parent  # Goes from notebook/ to RAG/
DATA_DIR = SCRIPT_DIR / "data"

'''
LOADING DOCUMENTS
'''
### Document Structure
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
doc = Document(
    page_content="This is the main text content I am using to create RAG system.",
    metadata={
        "source": "example.txt",
        "pages": 1,
        "author": "Soumick",
        "date_created": "2024-02-12"
        }
    )
# print(doc)

# Create a simplr text file
os.makedirs(DATA_DIR / "text_files", exist_ok=True)

sample_text = {
    str(DATA_DIR / "text_files/python_intro.txt"): """Python Programming Introduction
    
    Python is a high-level, interpreted programming language known for its simplicity and readability. 
    It was created by Guido van Rossum and first released in 1991. 
    Python supports multiple programming paradigms, including procedural, object-oriented, 
    and functional programming. It has a large standard library and 
    a vibrant ecosystem of third-party packages, making it a popular choice for 
    web development, data analysis, artificial intelligence, scientific computing, and more.
    
    Key features of Python include:
    - Easy to learn and use: Python's syntax is clear and concise, making it accessible to beginners.
    - Versatile: Python can be used for a wide range of applications, from web development to data science and machine learning.
    - Large community: Python has a vast and active community that contributes to its growth and development.
    - Extensive libraries: Python's standard library and third-party packages provide tools for various tasks,
    
    """,

    str(DATA_DIR / "text_files/machine_learning.txt"): """Machine Learning Overview
    
    Machine learning is a subset of artificial intelligence (AI) that focuses on enabling computers to learn and make decisions from data without being explicitly programmed. It involves the development of algorithms and statistical models that allow systems to improve their performance over time through experience.
    
    Key concepts in machine learning include:
    - Supervised Learning: Learning from labeled data.
    - Unsupervised Learning: Finding patterns in unlabeled data.
    - Reinforcement Learning: Learning through interaction with an environment.
    
    Applications of machine learning are widespread, including image recognition, natural language processing, recommendation systems, and autonomous vehicles.
    
    """,
}

for filepath, comtent in sample_text.items():
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(comtent)

# print("Sample text file created at '../data/text_files/python_intro.txt'")



### Text Loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader(str(DATA_DIR / "text_files/python_intro.txt"), encoding='utf-8')
data = loader.load()
# print(data) 


### Directory Loader
from langchain_community.document_loaders import DirectoryLoader
# load all text files in the directory
dir_loader = DirectoryLoader(
    str(DATA_DIR / "text_files"),
    glob="**/*.txt", #pattern to match all text files in the directory and subdirectories
    loader_cls=TextLoader, # specify the loader class to use for loading the files
    loader_kwargs={"encoding": "utf-8"}, # additional arguments to pass to the loader class
    show_progress=True # show progress bar while loading files
)
docs = dir_loader.load()
# print(docs)


### PDF Loader
from langchain_community.document_loaders import PyMuPDFLoader
# load all pdf files in the directory
dir_loader = DirectoryLoader(
    str(DATA_DIR / "pdf"),
    glob="**/*.pdf", #pattern to match all pdf files in the directory and subdirectories
    loader_cls=PyMuPDFLoader, # specify the loader class to use for loading the files
    show_progress=True # show progress bar while loading files
)
docs = dir_loader.load()
# print(docs)


'''
Chunking Documents
'''

def split_documents(documents,chunk_size=500,chunk_overlap=100):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

chunks=split_documents(docs)


'''
EMBEDDING
'''

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

"""
To avoid warnings like the below:

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

You can get your token from https://huggingface.co/settings/tokens
and then set it in your environment variables. For example, in a terminal you can run:
export HF_TOKEN='your_token_here'  # For Linux/Mac
set HF_TOKEN='your_token_here'  # For Windows

This is just for env injection through the terminal
"""

class EmbeddingManager:
    """Handles document embedding using SentenceTransformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Hugging Face model name for sentence embedding. Default is 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")  
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of strings to embed
        
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not self.model:
            raise ValueError("Model not loaded.")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    

## initialize embedding manager
embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')


'''
VECTOR STORE
'''

class VectorStore:
    """Manages document embeddings using ChromaDB vector store"""

    def __init__(self, collection_name: str = 'pdf_documents', persist_directory: str = '../data/vector_store'):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # Create persistent ChromaBD Client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of Langchain documents
            embeddings: corresponding embedding for the documents.
        """

        # Logic to prevent duplicate ingestion - check if collection already has data
        if self.collection.count() > 0:
            print("Collection already has data. Skipping ingestion to avoid duplicates.")
            return

        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents.")
        
        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata['doc_index'] = i
            metadata['context_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document Content
            documents_texts.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_texts
            )
            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Total documents in collection after addition: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

## initialize the vector store
vector_store = VectorStore()


## Convert the text to embeddings
texts = [doc.page_content for doc in chunks]

## Generate embeddings for the document chunks
embeddings = embedding_manager.generate_embeddings(texts) 

## Store in the vector store
vector_store.add_documents(chunks, embeddings)




'''
RAG RETRIEVAL PIPELINE FROM VECTOR STORE
'''

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    # similarity_score = 1 - distance  
                    
                    print(f"[DEBUG] Distance: {distance:.4f}")

                    # Filter based on score threshold
                    # if similarity_score >= score_threshold:
                    if distance < 1.1:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            #'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                        
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# initialize the retriever
rag_retriever=RAGRetriever(vector_store,embedding_manager)

# using a sample query to test the retrieval
sample_query = "What are the Python skills of Soumick Karmakar?"
retrieved_data = rag_retriever.retrieve(sample_query)
print("============================================================")
print(retrieved_data)
print("============================================================")




#***********************************#
#                                   #
#      DATA RETRIEVAL PIPELINE      #
#                                   #
#***********************************#


# RAG pipeline with Groq LLM
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

### Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)


### RAG Function: Retrieve Context + Generate Response
def rag_app(query, retriever, llm, top_k=3):
    """RAG pipeline to retrieve relevant documents and generate response"""

    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=top_k)
    if not retrieved_docs:
        return "No relevant information found in the documents."
    
    context = "\n\n".join([f"Document {doc['rank']}:\n{doc['content']}" for doc in retrieved_docs]) if retrieved_docs else ""
    if not retrieved_docs:
        return "No relevant information found in the documents."
    
    # Step 3: Generate response using LLM
    prompt = f"""
        Use the following context to answer the question concisely:
        Context: {context}

        Question: {query}

        Answer:
    """
    
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    return response.content


answer = rag_app(sample_query, rag_retriever, llm)
print("============================================================")
print(f"Query: {sample_query}")
print(f"Answer: {answer}")
print("============================================================")



# --- Enhanced RAG Pipeline Features ---
def advanced_rag(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        # 'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = 1 - min([doc.get('distance', 1.0) for doc in results])
    
    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

# Example usage:
result = advanced_rag("What are the Python skills of Soumick Karmakar?", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print("Answer:", result['answer'])
print("Sources:", result['sources'])
print("Confidence:", result['confidence'])
print("Context Preview:", result['context'][:300])
