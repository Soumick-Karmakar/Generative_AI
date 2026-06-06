#
# AGENTIC RAG WITH LANGGRAPH
#

from types import SimpleNamespace
from typing import TypedDict, List
import math
import re
import requests
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import os
from dotenv import load_dotenv
load_dotenv()


class GroqChat:
    """Small Groq chat wrapper with the same invoke shape used below."""

    def __init__(self, api_key: str, model: str, temperature: float = 0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def invoke(self, prompt: str):
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return SimpleNamespace(content=content)


class BagOfWordsEmbeddings(Embeddings):
    """Local embeddings for the small tutorial vector store."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * len(self.vocabulary)
        for token in self._tokenize(text):
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tokens = sorted({token for text in texts for token in self._tokenize(text)})
        self.vocabulary = {token: index for index, token in enumerate(tokens)}
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# Set GROQ_API_KEY in your .env file before running this script.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

# Initialize models
llm = GroqChat(
    api_key=groq_api_key,
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature=0,
)
embeddings = BagOfWordsEmbeddings()


# STATE DEFINITIONS
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


### Sample Docuemnt And VectorStore
sample_texts = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to provide context for generating more accurate responses.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They are commonly used in RAG systems to find relevant documents based on semantic similarity.",
    "Agentic systems are AI systems that can take actions, make decisions, and interact with their environment autonomously. They often use planning and reasoning capabilities."
]

documents=[Document(page_content=text) for text in sample_texts]

### create vector store
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



# AGENTS FUNCTIONS

def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide if we need to retrieve documents based on the question
    """
    question = state["question"]
    
    # Simple heuristic: if question contains certain keywords, retrieve
    retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]
    needs_retrieval = any(keyword in question.lower() for keyword in retrieval_keywords)
    
    return {**state, "needs_retrieval": needs_retrieval}



def retrieve_documents(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents based on the question
    """
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {**state, "documents": documents}



def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using the retrieved documents or direct response
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if documents:
        # RAG approach: use documents as context
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""Based on the following context, answer the question:

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
    else:
        # Direct response without retrieval
        prompt = f"Answer the following question: {question}"
    
    response = llm.invoke(prompt)
    answer = response.content
    
    return {**state, "answer": answer}


# CONDITIONAL LOGIC
def should_retrieve(state: AgentState) -> str:
    """
    Determine the next step based on retrieval decision
    """
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"
    


# BUILDING THE GRAPH

# Create the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Set entry point
workflow.set_entry_point("decide")

# Add conditional edges
workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()
#print(app)  #this will only print the object in terminals. To visualize the graph structure, we can use the draw_mermaid method provided by LangGraph.
print(app.get_graph().draw_mermaid())  #this will print the graph structure in mermaid format for visualization in terminals 



# TESTING THE AGENTIC SYSTEM
def ask_question(question: str):
    """
    Helper function to ask a question and get an answer
    """
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }
    
    result = app.invoke(initial_state)
    return result


# Test with a question that should trigger retrieval
question1 = "What is LangGraph?"
result1 = ask_question(question1)
print(result1)


# Test with another question
question2 = "How does RAG work?"
result2 = ask_question(question2)
print(f"Question: {question2}")
print(f"Retrieved documents: {len(result2['documents'])}")
print(f"Answer: {result2['answer']}")
print("\n" + "="*50 + "\n")
