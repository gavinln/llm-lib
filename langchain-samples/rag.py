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

import bs4
import fire
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


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


def rag_quickstart():
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = load_url_docs(url)
    splits = split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
            <context>
            {context}
            </context>

            Question: {question}
    """
    )
    question = "What is Task Decomposition?"

    retriever: VectorStoreRetriever = vectorstore.as_retriever()
    example_retrieval(retriever, question)

    # TODO: add format docs to the context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    out: str = rag_chain.invoke(question)
    print(out)
    vectorstore.delete_collection()


def rag_chat_history():
    pass


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
