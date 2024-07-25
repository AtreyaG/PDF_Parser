import os
import re
import shutil
from chromadb import Documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from utils import get_retriever


load_dotenv()



# FILE_PATH = './pdf_docs'
# VECTOR_DB_PATH = './vector_storage'
# DOC_STORE_PATH = './doc_storage'

FILE_PATH = os.getenv("INPUT_FILE_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH")



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{re.sub('\n', ' ', d.page_content)}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )


def main():
    docs = load_document()

    pretty_print_docs(docs)

    save_to_chroma(docs=docs)



def load_document():
    loader = PyPDFDirectoryLoader(FILE_PATH)
    documents = loader.load()


    return documents


def save_to_chroma(docs: list[Documents]):

    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    if os.path.exists(DOC_STORE_PATH):
        shutil.rmtree(DOC_STORE_PATH)

    # db = Chroma.from_documents(
    #     documents=docs, embedding=embedding_model, persist_directory=CHROMA_DB_PATH
    # )

    retriever = get_retriever()


    retriever.add_documents(docs)



if __name__ == "__main__":
    main()










