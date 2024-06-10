"""
https://docs.llamaindex.ai/en/latest/examples/vector_stores/DeepLakeIndexDemo/

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

from deeplake.core.dataset.dataset import Dataset
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data" / "paul_graham"))


def get_temp_storage_dir():
    return pathlib.Path(SCRIPT_DIR / "temp_storage")


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def main():
    documents = SimpleDirectoryReader(get_data_dir()).load_data()
    temp_storage_dir = get_temp_storage_dir()

    vector_store = DeepLakeVectorStore(
        dataset_path=str(temp_storage_dir), overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    print_query_response(query_engine, "What did the author learn?")

    print_query_response(
        query_engine, "What was a hard moment for the author?"
    )

    ds: Dataset = index.vector_store.client
    print("---size below and after deleting docs---")
    ds.summary()
    index.delete_ref_doc(documents[0].id_)
    ds.summary()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
