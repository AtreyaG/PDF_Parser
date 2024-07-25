from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate   
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

from utils import get_retriever
import re




embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")



PROMPT_TEMPLATE = (

    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use complete sentences to answer the question"
    "\n\n"
    "{context}"
)





prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("human", "{input}"),
    ]
)


def main():

    logging.basicConfig
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


    model = Ollama(model='llama3', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


    parent_child_retriever = get_retriever()

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=parent_child_retriever,
        llm=model
    )

    

    embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold = .7)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=multi_query_retriever
    )





    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    results = rag_chain.invoke(({"input": "What is the role of the Cultural Chair in the Chinese Student Association (CSA) at UT-Dallas?"}))




if __name__ == "__main__":
    main()