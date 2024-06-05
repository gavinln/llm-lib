"""
https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline/

OPENAI_API_KEY needed
"""

import logging
import pathlib

import fire

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline

# from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_dir_old():
    return str(pathlib.Path(SCRIPT_DIR / "data" / "paul_graham"))


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data"))


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


def get_pipeline():
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            OpenAIEmbedding(model="text-embedding-3-large"),
        ],
        docstore=SimpleDocumentStore(),
    )
    return pipeline


def pipeline_documents():
    print("ingestion pipeline docs")
    data_dir = get_data_dir()
    documents = SimpleDirectoryReader(
        data_dir, filename_as_id=True
    ).load_data()
    for doc in documents:
        print(doc)

    pipeline = get_pipeline()
    nodes = pipeline.run(documents=documents)
    for node in nodes:
        print(node.metadata)


def pipeline_redis():
    print("ingestion pipeline redis")


def main():
    fire.Fire(
        {
            "ingestion-pipeline-documents": pipeline_documents,
            "ingestion-pipeline-redis": pipeline_redis,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
