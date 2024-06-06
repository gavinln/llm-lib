"""
https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/

https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline/

OPENAI_API_KEY needed
"""

import logging
import pathlib
from typing import Any

import fire
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
    IngestionPipeline,
)

# from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data"))


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def get_pipeline() -> IngestionPipeline:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            OpenAIEmbedding(model="text-embedding-3-large"),
        ],
        docstore=SimpleDocumentStore(),
    )
    return pipeline


def get_documents(data_dir) -> list[Document]:
    documents = SimpleDirectoryReader(
        data_dir, filename_as_id=True
    ).load_data()
    return documents


def pipeline_documents():
    print("ingestion pipeline docs")
    data_dir = get_data_dir()
    documents = get_documents(data_dir)
    print("---------docs--------------")
    for doc in documents:
        print(doc)

    pipeline = get_pipeline()
    nodes = pipeline.run(documents=documents)
    print("---------nodes-------------")
    for node in nodes:
        print(node.metadata)


def get_schema() -> IndexSchema:
    custom_schema = IndexSchema.from_dict(
        {
            "index": {"name": "redis_vector_store", "prefix": "doc"},
            # customize fields that are indexed
            "fields": [
                # required fields for llamaindex
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                # custom vector field for bge-small-en-v1.5 embeddings
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": 3072,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )
    return custom_schema


def get_redis_pipeline(embed_model, custom_schema) -> IngestionPipeline:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ],
        docstore=RedisDocumentStore.from_host_and_port(
            "localhost", 6379, namespace="document_store"
        ),
        vector_store=RedisVectorStore(
            schema=custom_schema,
            redis_url="redis://localhost:6379",
        ),
        cache=IngestionCache(
            cache=RedisKVStore.from_host_and_port("localhost", 6379),
            collection="redis_cache",
        ),
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    return pipeline


def pipeline_redis():
    print("ingestion pipeline redis")
    data_dir = get_data_dir()
    documents = get_documents(data_dir)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    embeddings = embed_model.get_text_embedding(
        "Open AI new Embeddings models is great."
    )
    print("len(embeddings) = {}".format(len(embeddings)))

    custom_schema = get_schema()
    pipeline = get_redis_pipeline(embed_model, custom_schema)
    nodes = pipeline.run(documents=documents)
    print("---------nodes-------------")
    for node in nodes:
        print(node.metadata)

    index: VectorStoreIndex = VectorStoreIndex.from_vector_store(
        pipeline.vector_store, embed_model=embed_model  # type: ignore
    )
    query_engine = index.as_query_engine(similarity_top_k=10)
    print_query_response(query_engine, "What documents do you see?")

    vector_store: Any = pipeline.vector_store
    if vector_store.index_exists():
        vector_store.delete_index()


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
