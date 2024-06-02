"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/
"""

import logging
import pathlib

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.faiss import FaissVectorStore

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


def create_index_from_documents(data_dir, persist_dir) -> BaseIndex:
    dim = 1536
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def load_index_from_dir(persist_dir) -> BaseIndex:
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index


def main():
    print("faiss vector store")
    data_dir = get_data_dir()
    persist_dir = get_temp_storage_dir()
    _ = create_index_from_documents(data_dir, persist_dir)
    index = load_index_from_dir(persist_dir)
    query_engine = index.as_query_engine()
    print_query_response(query_engine, "What did the author do growing up?")
    print_query_response(
        query_engine, "What did the author do after his time at Y Combinator?"
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
