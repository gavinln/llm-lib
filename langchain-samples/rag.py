"""

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

rag-quickstart
https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/

rag-chat_history
https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

rag-streaming
https://python.langchain.com/v0.1/docs/use_cases/question_answering/streaming/

rag-citations
https://python.langchain.com/v0.1/docs/use_cases/question_answering/citations/
"""

import logging
import pathlib
import tempfile
from operator import itemgetter
from typing import List

import bs4
import fire
import wikipedia
from joblib import Memory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
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


def process_chunks(chain, question):
    output = {}
    curr_key = None
    for chunk in chain.stream(question):
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            if key != curr_key:
                print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
            else:
                print(chunk[key], end="", flush=True)
            curr_key = key


def rag_streaming():
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    vector_store = get_vector_store_docs_from_url(url)

    prompt = get_rag_prompt_tempate()

    retriever: VectorStoreRetriever = vector_store.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    question = "What is Task Decomposition?"
    for chunk in rag_chain_with_source.stream(question):
        print(chunk)
    process_chunks(rag_chain_with_source, question)
    vector_store.delete_collection()


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def print_docs_summary(docs):
    print(f"There are {len(docs)} documents")
    for doc in docs:
        print(f"{doc.dict()['metadata']['title']=}")


class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite
    the sources used."""

    answer: str = Field(
        description="The answer to the user question, which is based only "
        "on the given sources.",
    )
    citations: List[int] = Field(
        description="The integer IDs of the SPECIFIC sources which justify "
        "the answer.",
    )


def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"""Source ID: {i}\n
        Article Title: {doc.metadata['title']}\n
        Article Snippet: {doc.page_content}"""
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


def get_wikipedia_rag(llm, prompt, wiki_retriever, question):
    format = itemgetter("docs") | RunnableLambda(format_docs)
    # subchain for generating an answer once we've done retrieval
    answer = prompt | llm | StrOutputParser()
    # chain that calls wiki -> formats docs to string -> runs answer
    # -> returns just the answer and retrieved docs.
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=wiki_retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )
    out = chain.invoke(question)
    print(f"{out['answer']=}")
    print_docs_summary(out["docs"])


def get_wikipedia_rag_cited_documents(llm, prompt, wiki_retriever, question):
    llm_with_tool = llm.bind_tools(
        [cited_answer],
        tool_choice="cited_answer",
    )
    output_parser = JsonOutputKeyToolsParser(
        key_name="cited_answer", first_tool_only=True
    )
    format = itemgetter("docs") | RunnableLambda(format_docs_with_id)
    answer = prompt | llm_with_tool | output_parser
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=wiki_retriever)
        .assign(context=format)
        .assign(cited_answer=answer)
        .pick(["cited_answer", "docs"])
    )
    out = chain.invoke(question)
    print(f"{out['cited_answer']=}")
    print_docs_summary(out["docs"])


class Citation(BaseModel):
    source_id: int = Field(
        description="The integer ID of a SPECIFIC source which justifies "
        "the answer.",
    )
    quote: str = Field(
        description="The VERBATIM quote from the specified source that "
        "justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite
    the sources used."""

    answer: str = Field(
        description="The answer to the user question, which is based only "
        "on the given sources.",
    )
    citations: List[Citation] = Field(
        description="Citations from the given sources that justify the answer.",
    )


def get_wikipedia_rag_cited_snippets(llm, prompt, wiki_retriever, question):
    output_parser = JsonOutputKeyToolsParser(
        key_name="quoted_answer", first_tool_only=True
    )
    llm_with_tool = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )
    format = itemgetter("docs") | RunnableLambda(format_docs_with_id)
    answer = prompt | llm_with_tool | output_parser
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=wiki_retriever)
        .assign(context=format)
        .assign(quoted_answer=answer)
        .pick(["quoted_answer", "docs"])
    )
    out = chain.invoke(question)
    print(f"{out['quoted_answer']=}")
    print_docs_summary(out["docs"])


def rag_citations():
    question = "How fast are cheetahs?"
    wiki_retriever = WikipediaRetriever(
        wiki_client=wikipedia, top_k_results=6, doc_content_chars_max=2000
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a helpful AI assistant. Given a user question and some "
                "Wikipedia article snippets, answer the user question. If none "
                "of the articles answer the question, just say you don't know."
                "\n\nHere are the Wikipedia articles:{context}",
            ),
            ("human", "{question}"),
        ]
    )
    prompt.pretty_print()

    print("Get answer", "-" * 10)
    get_wikipedia_rag(llm, prompt, wiki_retriever, question)

    print("Cite documents", "-" * 10)
    get_wikipedia_rag_cited_documents(llm, prompt, wiki_retriever, question)

    print("Cite snippets", "-" * 10)
    get_wikipedia_rag_cited_snippets(llm, prompt, wiki_retriever, question)


def main():
    fire.Fire(
        {
            "rag-quickstart": rag_quickstart,
            "rag-chat-history": rag_chat_history,
            "rag-streaming": rag_streaming,
            "rag-citations": rag_citations,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
