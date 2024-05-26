"""
https://platform.openai.com/docs/guides/embeddings
"""

import logging
import pathlib

import fire
from llama_index.core import (SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.redis import RedisVectorStore
from redis import Redis

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_default_vector_store_index(persist_dir) -> BaseIndex:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index


def persist_default_vector_store_index(persist_dir, documents) -> BaseIndex:
    index: BaseIndex = VectorStoreIndex.from_documents(
        documents,
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def default_vector_store():
    print("using default vector store")
    persist_dir = pathlib.Path(SCRIPT_DIR / "temp_storage")
    if persist_dir.exists():
        index = get_default_vector_store_index(persist_dir)
    else:
        documents = SimpleDirectoryReader("data").load_data()
        index = persist_default_vector_store_index(persist_dir, documents)

    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)


def get_redis_vector_store_index(url) -> BaseIndex:
    redis_client = Redis.from_url(url)
    vector_store = RedisVectorStore(redis_client=redis_client, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def persist_redis_vector_store_index(url, documents) -> BaseIndex:
    redis_client = Redis.from_url(url)
    vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index: BaseIndex = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


def next_exists(iterable):
    if next(iterable, None) is None:
        return False
    return True


def redis_vector_store():
    print("using redis vector store")
    url = "redis://localhost:6379"
    redis_client = Redis.from_url(url)
    assert redis_client.ping() is True, "Cannot connect to Redis"
    if next_exists(redis_client.scan_iter("llama_index/*")):
        index = get_redis_vector_store_index(url)
    else:
        documents = SimpleDirectoryReader("data").load_data()
        index = persist_redis_vector_store_index(url, documents)

    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "default-vector-store": default_vector_store,
            "redis-vector-store": redis_vector_store,
        }
    )


if __name__ == "__main__":
    main()
