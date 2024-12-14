import pickle

from typing import List
from typing_extensions import TypedDict

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langchain_community.document_loaders import PyPDFLoader

from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import EMBEDDING_MODEL, LARGE_LANGUAGE_MODEL,RETRIEVE_N_CHUNKS


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatMistralAI(model=LARGE_LANGUAGE_MODEL)

vector_store = InMemoryVectorStore(embeddings)
with open("./data/vector_store.pkl", "rb") as file:
    vector_store = pickle.load(file)


class StateSchema(TypedDict):
    messages: List[str]
    documents: List[str]
    document_text: str
    query: str
graph_builder = StateGraph(StateSchema)

def retrieve(state):
    print(state["messages"])
    query = state["messages"][-1]["content"]
    retrieved_docs = vector_store.similarity_search(query, k=RETRIEVE_N_CHUNKS)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return {"messages": state["messages"], "documents": retrieved_docs, "document_text": serialized, "query": query}


def generate(state):
    docs_content = state["document_text"]
    system_message_content = (
        "You are an assistant for question-answering tasks related to German laws. "
        "You will receive a question and content related to that question. Use the content to answer the question." 
        "\nIF the content is not relevant to the question then say no relevant dcument found"
        "\nIf you don't know the answer, say that you don't know. "
        "\n\n"
    )
    history = "Past messages:" + str([f"Query:{msg['content']}; Answer:{msg['content']}" for msg in state['messages'][:-1]])
    user_message = f"\n\nUser Query:{state['query']}\nRelevant Content:{state['document_text']}"
    extras = "\nUse three sentences maximum and keep the answer concise. DO NOT HALLUCINATE. ALWAYS RESPOND IN THE LANGUAGE OF THE USER"
    prompt = system_message_content + history + user_message + extras
    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(retrieve)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)


GRAPH = graph_builder.compile()
