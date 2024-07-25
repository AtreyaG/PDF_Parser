
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_retriever():
    fs = LocalFileStore(DOC_STORE_PATH)
    store = create_kv_docstore(fs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size = 256)

    vectorstore = Chroma(collection_name="split_parents", embedding_function=embedding_model, persist_directory= VECTOR_DB_PATH)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity", 
        search_kwargs={"k": 2}
    )

    return retriever


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    return embedding_model





