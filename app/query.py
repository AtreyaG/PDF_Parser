from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage


from utils import get_retriever, get_embedding_model, get_chat_model, get_llm_model
from templates import QUERY_PROMPT, SYSTEM_PROMPT, CONTEXTUALIZE_PROMPT


chat_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),

    ]
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chat_llm = get_chat_model()
llm = get_llm_model()


def create_multi_query_retreiver():

    parent_child_retriever = get_retriever()

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=parent_child_retriever,
        llm=chat_llm,
        prompt=QUERY_PROMPT
    )

    return multi_query_retriever


def create_compressor():

    _filter = LLMChainFilter.from_llm(llm)

    multi_query_retriever = create_multi_query_retreiver()

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=multi_query_retriever
    )

    return compression_retriever



def main():


    compression_retriever = create_compressor()

   
    retriever_chain = create_history_aware_retriever(llm, compression_retriever, chat_history_prompt)
    question_answer_chain = create_stuff_documents_chain(chat_llm, prompt)
    rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)



    chat_history = []



    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        chat_history.append(HumanMessage(content=query))

        results = rag_chain.invoke(({
            "chat_history": chat_history,
            "input": query 
        }))

        print("\n")

        chat_history.append(AIMessage(content=results['answer']))







if __name__ == "__main__":
    main()