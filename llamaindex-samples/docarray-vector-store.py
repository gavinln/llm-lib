"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/DocArrayInMemoryIndexDemo/

OPENAI_API_KEY needed
"""

import logging
import pathlib
from typing import Any

import fire
from llama_index.core import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.docarray import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)

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


def create_index_from_vector_store(vector_store, data_dir) -> BaseIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(data_dir).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


def create_index_from_nodes(nodes, vector_store) -> BaseIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
    return index


def get_movie_nodes() -> list[TextNode]:
    nodes = [
        TextNode(
            text="The Shawshank Redemption",
            metadata={  # type: ignore
                "author": "Stephen King",
                "theme": "Friendship",
            },
        ),
        TextNode(
            text="The Godfather",
            metadata={  # type: ignore
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
        ),
        TextNode(
            text="Inception",
            metadata={  # type: ignore
                "director": "Christopher Nolan",
            },
        ),
    ]
    return nodes


def inmemory_vector_store():
    """
    MetaDataFilters does not work
    """
    data_dir = get_data_dir()
    vector_store: Any = DocArrayInMemoryVectorStore()
    index = create_index_from_vector_store(vector_store, data_dir)

    query_engine = index.as_query_engine()
    query = "What did the author do growing up?"
    print_query_response(query_engine, query)

    query = "What was a hard moment for the author?"
    print_query_response(query_engine, query)

    nodes = get_movie_nodes()
    index = create_index_from_nodes(nodes, vector_store)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="theme", value="Mafia")]
    )
    retriever = index.as_retriever(filters=filters)
    print(retriever)
    # print_query_nodes(retriever, "What is inception about?")


def hnsw_vector_store():
    """
    DOES NOT WORK

    ValueError: Search field embedding is not present in the HNSW indices
    """
    data_dir = get_data_dir()
    temp_dir = get_temp_storage_dir()
    vector_store: Any = DocArrayHnswVectorStore(work_dir=str(temp_dir))
    index = create_index_from_vector_store(vector_store, data_dir)

    query_engine = index.as_query_engine()
    query = "What did the author do growing up?"
    print(query_engine, query)
    # print_query_response(query_engine, query)

    # query = "What was a hard moment for the author?"
    # print_query_response(query_engine, query)

    nodes = get_movie_nodes()
    index = create_index_from_nodes(nodes, vector_store)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="theme", value="Mafia")]
    )
    retriever = index.as_retriever(filters=filters)
    print(retriever)
    # print_query_nodes(retriever, "What is inception about?")


def main():
    fire.Fire(
        {
            "docarray-inmemory-vector": inmemory_vector_store,
            "docarray-hnsw-vector": hnsw_vector_store,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
