'''
DATA INGESTION PIPELINE
'''


'''
LOADING DOCUMENTS
'''
### Document Structure
from langchain_core.documents import Document
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
import os
os.makedirs('../data/text_files/', exist_ok=True)

sample_text = {
    "../data/text_files/python_intro.txt": """Python Programming Introduction
    
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

    "../data/text_files/machine_learning.txt": """Machine Learning Overview
    
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
loader = TextLoader('../data/text_files/python_intro.txt', encoding='utf-8')
data = loader.load()
# print(data) 


### Directory Loader
from langchain_community.document_loaders import DirectoryLoader
# load all text files in the directory
dir_loader = DirectoryLoader(
    "../data/text_files/",
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
    "../data/pdf/",
    glob="**/*.pdf", #pattern to match all pdf files in the directory and subdirectories
    loader_cls=PyMuPDFLoader, # specify the loader class to use for loading the files
    show_progress=True # show progress bar while loading files
)
docs = dir_loader.load()
# print(docs)


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
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if not self.model:
            raise ValueError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()

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
