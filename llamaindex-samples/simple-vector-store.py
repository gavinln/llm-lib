"""
svm_regression

https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemo/

async_index

https://docs.llamaindex.ai/en/stable/examples/vector_stores/AsyncIndexCreationDemo/

maximal_marginal_rel

https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoMMR/
"""

import logging
import pathlib
import tempfile

import fire
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


def svm_regression():
    temp_storage_dir = get_temp_storage_dir()
    # index = get_default_vector_store_index(temp_storage_dir)
    data_dir = get_data_dir()
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = persist_default_vector_store_index(temp_storage_dir, documents)
    query = "What did the author do growing up?"

    query_modes = [
        "svm",
        "linear_regression",
        "logistic_regression",
    ]
    for query_mode in query_modes:
        query_engine = index.as_query_engine(
            vector_store_query_mode=query_mode
        )
        print(f"--{query_mode=}")
        print_query_response(query_engine, query)
    # custom embedding string
    query_bundle = QueryBundle(
        query_str=query,
        custom_embedding_strs=["The author grew up painting."],
    )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_bundle)
    print(f"There are {len(response.source_nodes)} source nodes")

    # add tagged doc and query index with filters
    doc = Document(text="target", metadata={"tag": "target"})  # type: ignore
    index.insert(doc)
    retriever = create_retriever(index)
    print_query_nodes(retriever, query)


@memory.cache
def get_wikipedia_documents(pages):
    "download wikipedia documents - about 6 seconds"
    loader = WikipediaReader()
    documents = loader.load_data(pages=pages)
    return documents


def get_wikipedia_pages():
    pages = [
        "Berlin",
        "Santiago",
        "Moscow",
        "Tokyo",
        "Jakarta",
        "Cairo",
        "Bogota",
        "Shanghai",
        "Damascus",
    ]
    return pages


@memory.cache
def create_index_from_documents(documents) -> BaseIndex:
    index = VectorStoreIndex.from_documents(documents, use_async=True)
    return index


def async_index():
    pages = get_wikipedia_pages()
    docs = get_wikipedia_documents(pages)
    index = create_index_from_documents(docs)
    query_engine = index.as_query_engine()  # type: ignore
    query = "What is the etymology of Jakarta"
    print_query_response(query_engine, query)


def maximum_marginal_rel():
    temp_storage_dir = get_temp_storage_dir()
    data_dir = get_data_dir()
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = persist_default_vector_store_index(temp_storage_dir, documents)
    retriever = index.as_retriever(
        vector_store_query_mode="mmr",
        similarity_top_k=3,
        vector_store_kwargs={"mmr_threshold": 0.7},
    )
    query = "What did the author do during his time in Y Combinator?"
    print_query_nodes(retriever, query)
    # breakpoint()


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "vector-svm-regression": svm_regression,
            "vector-async-index": async_index,
            "vector-maximum-marginal-rel": maximum_marginal_rel,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
