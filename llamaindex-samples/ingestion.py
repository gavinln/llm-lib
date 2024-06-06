"""
https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/

https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline/

OPENAI_API_KEY needed
"""

import logging
import pathlib
from typing import Any

import faiss
import fire
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.faiss import FaissVectorStore
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


def get_pipeline_documents() -> IngestionPipeline:
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
    data_dir = get_data_dir()
    documents = get_documents(data_dir)
    print("---------docs--------------")
    for doc in documents:
        print(doc)

    pipeline = get_pipeline_documents()
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


def get_doc_sample():
    doc = """
    Context

    LLMs are a phenomenal piece of technology for knowledge generation and
    reasoning. They are pre-trained on large amounts of publicly available
    data. How do we best augment LLMs with our own private data? We need a
    comprehensive toolkit to help perform this data augmentation for LLMs.

    Proposed Solution

    That's where LlamaIndex comes in. LlamaIndex is a "data framework" to help
    you build LLM  apps. It provides the following tools:

    Offers data connectors to ingest your existing data sources and data
    formats (APIs, PDFs, docs, SQL, etc.)

    Provides ways to structure your data (indices, graphs) so that this data
    can be easily used with LLMs.

    Provides an advanced retrieval/query interface over your data:

    Feed in any LLM input prompt, get back retrieved context and
    knowledge-augmented output.

    Allows easy integrations with your outer application framework (e.g. with
    LangChain, Flask, Docker, ChatGPT, anything else).

    LlamaIndex provides tools for both beginner users and advanced users.

    Our high-level API allows beginner users to use LlamaIndex to ingest and
    query their data in 5 lines of code.

    Our lower-level APIs allow advanced users to customize and extend any
    module (data connectors, indices, retrievers, query engines, reranking
    modules), to fit their needs.

    """
    return doc


def get_pipeline(vector_store: BasePydanticVectorStore) -> IngestionPipeline:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=50, chunk_overlap=0),
            TitleExtractor(),
            OpenAIEmbedding(model="text-embedding-3-large"),
        ],
        vector_store=vector_store,
    )
    return pipeline


def pipeline():
    dim = 3072
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    doc = Document(text=get_doc_sample())
    pipeline = get_pipeline(vector_store)
    nodes = pipeline.run(documents=[doc])
    print("---------nodes-------------")
    for node in nodes:
        print(node.metadata)


def main():
    fire.Fire(
        {
            "ingestion-pipeline": pipeline,
            "ingestion-pipeline-documents": pipeline_documents,
            "ingestion-pipeline-redis": pipeline_redis,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
