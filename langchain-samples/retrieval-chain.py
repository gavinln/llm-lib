"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib
import tempfile
from typing import Any

from bs4 import BeautifulSoup
from joblib import Memory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)


def example_html_doc():
    html_doc = """
    <html><head><title>The Dormouse's story</title></head>
    <body>
    <p class="title"><b>The Dormouse's story</b></p>

    <p class="story">Once upon a time there were three little
    sisters; and their names were
    <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
    <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
    <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>

    <p class="story">...</p>
    """
    return html_doc


def get_soup_example(html_doc: str):
    soup = BeautifulSoup(html_doc, "html.parser")
    return soup


def beautifulsoup_example():
    html_doc = example_html_doc()
    soup: Any = get_soup_example(html_doc)
    print(f"{soup.title=}")
    print(f"{soup.title.name=}")
    print(f"{soup.title.string=}")
    print(f"{soup.title.parent.name=}")
    print(f"{soup.find_all('a')=}")

    print("----get all links----------------")
    for link in soup.find_all("a"):
        print(link.get("href"))

    print("----get text----------------")
    print(soup.get_text())

    print("----prettify----------------")
    print(soup.prettify())


@memory.cache
def llm_query_no_rag(query: str):
    llm = ChatOpenAI()
    query = "how can langsmith help with testing?"
    return llm.invoke(query)


@memory.cache
def get_web_documents(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


@memory.cache
def get_split_documents(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter()
    print("original docs length: {:>12,d}".format(len(docs[0].page_content)))
    documents = text_splitter.split_documents(docs)
    for doc in documents:
        print("   split docs length: {:>12,d}".format(len(doc.page_content)))
    return documents


def get_docs_vector(docs: list[Document]):
    documents = get_split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def example_document_chain_invoke(document_chain: Any, query: str):
    message: str = document_chain.invoke(
        {
            "input": query,
            "context": [
                Document(
                    page_content="langsmith can let you visualize test results"
                )
            ],
        }
    )
    print(message)


def main():
    # beautifulsoup_example()
    llm = ChatOpenAI()
    query = "How can langsmith help with testing? Be concise."
    message = llm_query_no_rag(query)
    print("llm_query_no_rag", "-" * 10)
    print(message.content)

    url = "https://docs.smith.langchain.com/user_guide"
    docs = get_web_documents(url)
    assert len(docs) > 0, "Cannot get documents"
    vector_store = get_docs_vector(docs)

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
            <context>
            {context}
            </context>

            Question: {input}
    """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    # example_document_chain_invoke(document_chain, query)

    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response_dict = retrieval_chain.invoke({"input": query})
    print(f"{response_dict['input']=}")
    print(f"{response_dict['answer']=}")
    for idx, context in enumerate(response_dict["context"]):
        print(f"context {idx}", "-" * 10)
        print("context length: {:>12,d}".format(len(context.page_content)))

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query "
                + "to look up to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    chat_history = [
        HumanMessage(
            content="Can LangSmith help test my LLM applications? Be concise."
        ),
        AIMessage(content="Yes!"),
    ]
    response_list = retriever_chain.invoke(
        {"chat_history": chat_history, "input": "Tell me how. Be concise."}
    )
    for idx, response in enumerate(response_list):
        print(f"response {idx}")
        print(f"response length {len(response.page_content)}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
