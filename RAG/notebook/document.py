'''
DATA INGESTION PIPELINE
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
print(doc)
