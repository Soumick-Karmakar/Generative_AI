'''
RAG : Retrieval-Augmented Generation
------------------------------------------
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, 
so it references an authoritative knowledge base outside of its training data sources 
before generating a response. Large Language Models (LLMs) are trained on vast volumes of data 
and use billions of parameters to generate original output for tasks like answering questions, 
translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs 
to specific domains or an organization's internal knowledge base, all without the need 
to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, 
accurate, and useful in various contexts.
----------------------------------------------

RAG will be implemented using LangChain. 

So to initialize the LangChain, we need to run the command 'uv init' in the terminal. 
This will create a 'uv.lock' file in the project directory, 
which is used to manage dependencies and ensure that the correct versions of packages are installed.

Basic steps to set up the environment for RAG using LangChain:
1. uv init
2. uv venv
3. source .venv/Scripts/activate
4. Create a 'requirements.txt' file and add the following dependencies:
    - langchain
    - langchain-core
    - langchain-community
5. uv add -r requirements.txt

'''

def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
