#######[ RAG with Mongo DB Data Ingestion, Retrieval and Generation Pipeline ]#######

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"


##Function to generate embeddings
def get_embedding(text, input_type="document"):
    task_type = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values

embed = get_embedding("This is a sample document to generate embedding.")
#print(embed)




#####################[ Data Ingestion ]######################

"""Document Loading and Preprocessing"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("https://investors.mongodb.com/node/12236/pdf")
data = loader.load()

## split the data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
documents = text_splitter.split_documents(data)

# Prepare the data for MongoDB insertion
docs_to_insert = [
    {
        'text': doc.page_content,
        'embedding': get_embedding(doc.page_content)
    } for doc in documents
]

## MongoDB Connection and Data Insertion
from pymongo import MongoClient

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster_url = os.getenv("MONGO_CLUSTER_URL")
client=MongoClient(f"mongodb+srv://{username}:{password}@{cluster_url}/?appName=RAG")
collection=client['sample_mflix']['RAG_Docs']

# Insert the documents into MongoDB
result = collection.insert_many(docs_to_insert)




#####################[ Data Retrieval ]######################

"""Performing the Vector Search Query"""

## Query with Search Index
from pymongo.operations import SearchIndexModel
import time

# Create a search index on the 'embedding' field
vector_index = "vector_index"
search_index_modal = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "numDimensions": len(docs_to_insert[0]['embedding']),
                "path": "embedding",
                "similarity": "cosine"
            }
        ]
    },
    name=vector_index,
    type = "vectorSearch"
)

# This is just to create the index before running the search query, you can comment this out after the index is created once
collection.create_search_index(model = search_index_modal)   

query_embeddings = get_embedding("Vector search in mongo db")


results = collection.aggregate([
    {
        "$vectorSearch": {
            "index": vector_index,
            "path": "embedding",
            "queryVector": query_embeddings,
            "numCandidates": min(len(docs_to_insert), 100),
            "limit": 5
        }
    }
])

array_of_results = []
for doc in results:
    array_of_results.append(doc)
# print(array_of_results)

# Function to run vector search queries
def get_query_results(query):
    query_embeddings = get_embedding(query, input_type="query")
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": "embedding",
                "queryVector": query_embeddings,
                "numCandidates": min(len(docs_to_insert), 100),
                "limit": 5
            }
        },  
        {
            "$project": {
                "_id": 0,
                "text": 1
            }
        }
    ]

    results = collection.aggregate(pipeline)
    # print(list(results))

    array_of_results = []
    for docs in results:
        array_of_results.append(docs)
    
    return array_of_results

query_results = get_query_results("What do you know about MongoDB Investors?")



"""
Currently the abstraction is done manually:

collection.aggregate([
   {"$vectorSearch": ...}
])

However, in real AI/ML Engineer interviews and production systems the following combinations are used for Vector Store Abstraction:

LangChain + MongoDB Atlas Vector Search
LangChain + FAISS
LangChain + Chroma
LangChain + Pinecone
LangChain + Weaviate

"""


#####################[ Data Generation ]######################

"""LLM Integration and Prompting"""

# Specifying search qury, retrieving relevant documents and converting to string
query = "What do you know about MongoDB Investors?"
context_docs = get_query_results(query)
context_string = " ".join([doc['text'] for doc in context_docs])


# Construct prompt for the LLM using the relevant docs as the context
prompt = f"""Use the following pieces of context to answer the question at the end: 
            Context: {context_string} 
            Question: {query}
        """

print("Number of retrieved docs:", len(context_docs))
print("Context length:", len(context_string))
print("Prompt length:", len(prompt))
print(context_docs[:2])
print("=" * 50)
print("Retrieved docs:", len(context_docs))
print("Context length:", len(context_string))
print("Prompt length:", len(prompt))
print("=" * 50)
print(collection.count_documents({}))
print(prompt[:1000])
print("Using model:", GENERATION_MODEL)

completion = gemini_client.models.generate_content(
    model=GENERATION_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant for answering questions about MongoDB Vector Search."
    ),
)

print(completion.text)
