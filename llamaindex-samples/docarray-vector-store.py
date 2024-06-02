"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/DocArrayInMemoryIndexDemo/
"""

import logging
import pathlib
from typing import Any

from llama_index.core import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data" / "paul_graham"))


def get_temp_storage_dir():
    return pathlib.Path(SCRIPT_DIR / "temp_storage")


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def print_query_nodes(retriever: BaseRetriever, query):
    result_nodes = retriever.retrieve(query)
    print(f"--{query}----")
    no_nodes = True
    for node in result_nodes:
        print(node)
        no_nodes = False
    if no_nodes:
        print("There are NO NODES")


def create_index_from_documents(data_dir) -> BaseIndex:
    vector_store: Any = DocArrayInMemoryVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(data_dir).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


def main():
    print("docarray vector store")
    data_dir = get_data_dir()
    index = create_index_from_documents(data_dir)
    query_engine = index.as_query_engine()
    query = "What did the author do growing up?"
    print_query_response(query_engine, query)

    query = "What was a hard moment for the author?"
    print_query_response(query_engine, query)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
