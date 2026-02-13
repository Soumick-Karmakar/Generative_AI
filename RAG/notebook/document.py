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

print("Sample text file created at '../data/text_files/python_intro.txt'")



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
print(docs)