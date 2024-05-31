"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/
"""

import logging
import pathlib
import tempfile

from joblib import Memory
from llama_index.core import (
    Document,
    QueryBundle,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.readers.wikipedia import WikipediaReader

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)


def get_default_vector_store_index(persist_dir) -> BaseIndex:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index


def persist_default_vector_store_index(persist_dir, documents) -> BaseIndex:
    index: BaseIndex = VectorStoreIndex.from_documents(
        documents,
    )
    index.set_index_id("vector_index")
    index.storage_context.persist(persist_dir=persist_dir)
    return index


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


def create_retriever(index: BaseIndex) -> BaseRetriever:
    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(key="tag", value="target"),
            ],
        ),
    )
    return retriever


def create_index_from_documents(documents) -> BaseIndex:
    index = VectorStoreIndex.from_documents(documents, use_async=True)
    return index


def main():
    print("faiss vector store")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
