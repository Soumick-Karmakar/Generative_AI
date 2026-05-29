import typesense
import os
from dotenv import load_dotenv

load_dotenv()


client=typesense.Client({
  'nodes': [{
    'host': 'xfsd62iahlz1vju7p-1.a2.typesense.net', # in case of local host use 'localhost'
    'port': '443',                                  # for local use :8000, or similar ports
    'protocol': 'https'                             # for local, use 'http' 
  }],
  'api_key': os.getenv('Typesense_API_KEY'),
  'connection_timeout_seconds': 2
})


# Create the collection schema if it doesn't exist
schema = {
    "name": "books",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "title", "type": "string", "infix": True},
        {"name": "authors", "type": "string[]"},
        {"name": "publication_year", "type": "int32"},
        {"name": "average_rating", "type": "float"},
        {"name": "ratings_count", "type": "int32"}
    ],
    "default_sorting_field": "ratings_count"
}

try:
    client.collections.create(schema)
    print("Collection 'books' created successfully")
except Exception as e:
    print(f"Collection already exists or error: {e}")


with open('books.jsonl', 'r', encoding='utf-8') as jsonl_file:
    data = jsonl_file.read()
    client.collections['books'].documents.import_(data)


# search_parameters are listed below.
search_parameters={
    'q':"harry potter",
    'query_by':"title,authors",
    'sort_by':"ratings_count:desc"
}

# search_parameters = {
#   'q'         : 'harry potter',
#   'query_by'  : 'title',
#   'filter_by' : 'publication_year:<1998',
#   'sort_by'   : 'publication_year:desc'
# }

# search_parameters = {
#   'q'         : 'experiment',
#   'query_by'  : 'title',
#   'facet_by'  : 'authors',
#   'sort_by'   : 'average_rating:desc'
# }

result = client.collections['books'].documents.search(search_parameters)
# print(result)



########## Langchain + Typsense + Groq LLM + RAG Application ##########

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Typesense
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq


loader = TextLoader("test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings()
docsearch=Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": "xfsd62iahlz1vju7p-1.a2.typesense.net",  # Use xxx.a1.typesense.net for Typesense Cloud
        "port": "443",  # Use 443 for Typesense Cloud
        "protocol": "https",  # Use https for Typesense Cloud
        "typesense_api_key":os.getenv('Typesense_API_KEY'),
        "typesense_collection_name": "lang-chain"
    },
)


query = "What is artificial intelligence"
found_docs = docsearch.similarity_search(query)
print(found_docs[0].page_content)


### Retriever
retriever = docsearch.as_retriever()
retriever


query = "Artificial intelligence indepth explanation"
retriever.invoke(query)[0]




