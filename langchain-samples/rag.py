"""
https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/

A RAG (Retrieval augmented generation) application involves two stages

1. Indexing
2. Retrieval & generation

The indexing stage has the following steps

a. Load the documents
b. Split documents into smaller parts
c. Store the split parts into a vector store in the form of embeddings

The retrieval and generation part is composed of two steps

a. Retrieve the relevant split document parts
b. Generate the answer with an LLM using a prompt and the retrieved data
"""

import logging
import pathlib
import tempfile

import bs4
import fire
from joblib import Memory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableBinding
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


memory = Memory(tempfile.gettempdir(), verbose=0)


@memory.cache
def load_url_docs(url: str) -> list[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs: list[Document] = loader.load()
    return docs


@memory.cache
def split_documents(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    return splits


def example_retrieval(retriever: VectorStoreRetriever, question: str):
    retrieved_docs = retriever.invoke(question)
    for idx, doc in enumerate(retrieved_docs):
        print(f"retrieved doc {idx}", "-" * 10)
        print(doc.page_content[:75])


def get_vector_store_docs_from_url(url: str) -> Chroma:
    docs = load_url_docs(url)
    splits = split_documents(docs)
    vector_store: Chroma = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )
    return vector_store


def get_rag_prompt_tempate() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
            <context>
            {context}
            </context>

            Question: {question}
    """
    )
    return prompt


def rag_quickstart():
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    vector_store = get_vector_store_docs_from_url(url)
    prompt = get_rag_prompt_tempate()
    question = "What is Task Decomposition?"

    retriever: VectorStoreRetriever = vector_store.as_retriever()
    example_retrieval(retriever, question)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    out: str = rag_chain.invoke(question)
    print(out)
    vector_store.delete_collection()


def print_documents_start(documents: list[Document]):
    for idx, doc in enumerate(documents):
        print(f"doc {idx}", "-" * 10)
        print(doc.page_content[:75])


def rag_chain_example(rag_chain: RunnableBinding, question: str):
    chat_history = []

    ai_msg_1 = rag_chain.invoke(
        {"input": question, "chat_history": chat_history}
    )
    chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

    second_question = "What are common ways of doing it?"
    ai_msg_2 = rag_chain.invoke(
        {"input": second_question, "chat_history": chat_history}
    )
    print(ai_msg_2["answer"])
    return ai_msg_2["context"]


def rag_chat_history():
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    vector_store = get_vector_store_docs_from_url(url)
    retriever: VectorStoreRetriever = vector_store.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the \
    latest user question which might reference context in the chat \
    history, formulate a standalone question which can be understood \
    without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    question = "What is Task Decomposition?"

    context_documents = rag_chain_example(rag_chain, question)
    print_documents_start(context_documents)

    store = {}  # chat history

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("question 1", "-" * 10)
    out1 = conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    print(f"{out1['input']=}")
    print(f"{out1['answer']=}")

    print("question 2", "-" * 10)
    out2 = conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(f"{out2['input']=}")
    print(f"{out2['answer']=}")
    vector_store.delete_collection()


def rag_streaming():
    pass


def rag_sources():
    pass


def rag_citations():
    pass


def main():
    fire.Fire(
        {
            "rag-quickstart": rag_quickstart,
            "rag-chat-history": rag_chat_history,
            "rag-streaming": rag_streaming,
            "rag-sources": rag_sources,
            "rag-citations": rag_citations,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
