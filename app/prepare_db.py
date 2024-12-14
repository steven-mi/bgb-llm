import pickle
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import EMBEDDING_MODEL, LARGE_LANGUAGE_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)

loader = PyPDFLoader("./data/bgb.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(documents=all_splits)

with open("./data/vector_store.pkl", "wb") as file:
    pickle.dump(vector_store, file)
