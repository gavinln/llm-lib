"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/RedisIndexDemo/
"""

import datetime
import logging
import pathlib
import sys
from dataclasses import make_dataclass
from typing import Any

import fire
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.redis import RedisVectorStore
from redis import Redis
from redisvl.schema import IndexSchema

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


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data" / "paul_graham"))


def default_vector_store():
    print("using default vector store")
    persist_dir = pathlib.Path(SCRIPT_DIR / "temp_storage")
    if persist_dir.exists():
        index = get_default_vector_store_index(persist_dir)
    else:
        documents = SimpleDirectoryReader(get_data_dir()).load_data()
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


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def print_query_nodes(retriever: BaseRetriever, query):
    result_nodes = retriever.retrieve(query)
    print(f"--{query}----")
    for node in result_nodes:
        print(node)


def redis_vector_store():
    url = "redis://localhost:6379"
    redis_client = Redis.from_url(url)
    assert redis_client.ping() is True, "Cannot connect to Redis"
    if next_exists(redis_client.scan_iter("llama_index/*")):
        index = get_redis_vector_store_index(url)
        sys.exit("Index exists. Exiting")
    else:
        documents = SimpleDirectoryReader(get_data_dir()).load_data()
        index = persist_redis_vector_store_index(url, documents)

    query_engine = index.as_query_engine()
    query = "What did the author do growing up?"
    print_query_response(query_engine, query)

    retriever = index.as_retriever()
    query = "What did the author learn?"
    print_query_nodes(retriever, query)
    print_query_response(query_engine, query)

    query = "What was a hard moment for the author?"
    print_query_nodes(retriever, query)
    print_query_response(query_engine, query)


def get_custom_schema():
    custom_schema = IndexSchema.from_dict(
        {
            # customize basic index specs
            "index": {
                "name": "paul_graham",
                "prefix": "essay",
                "key_separator": ":",
            },
            # customize fields that are indexed
            "fields": [
                # required fields for llamaindex
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                # custom metadata fields
                # updated_at is a timestamp
                {"type": "numeric", "name": "updated_at"},
                {"type": "tag", "name": "file_name"},
                # custom vector field definition for cohere embeddings
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": 1536,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )
    return custom_schema


def persist_redis_vector_store_index_schema(
    url, documents, schema
) -> BaseIndex:
    redis_client = Redis.from_url(url)
    vector_store = RedisVectorStore(
        schema=schema, redis_client=redis_client, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index: BaseIndex = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


def date_to_timestamp(date_string: str) -> int:
    date_format: str = "%Y-%m-%d"
    return int(
        datetime.datetime.strptime(date_string, date_format).timestamp()
    )


def create_retriever(index: BaseIndex, updated_at: int) -> BaseRetriever:
    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(
                    key="file_name", value="paul_graham_essay.txt"
                ),
                MetadataFilter(
                    key="updated_at",
                    value=updated_at,
                    operator=">=",  # type: ignore
                ),
                MetadataFilter(
                    key="text",
                    value="learn",
                    operator="text_match",  # type: ignore
                ),
            ],
            condition="and",
        ),
    )
    return retriever


def print_keys(redis_client: Redis):
    items = redis_client.keys("*")
    print(items)


redis_index_info = """
    index_name
    index_options
    index_definition
    attributes
    num_docs
    max_doc_id
    num_terms
    num_records
    inverted_sz_mb
    vector_index_sz_mb
    total_inverted_index_blocks
    offset_vectors_sz_mb
    doc_table_size_mb
    sortable_values_size_mb
    key_table_size_mb
    geoshapes_sz_mb
    records_per_doc_avg
    bytes_per_record_avg
    offsets_per_term_avg
    offset_bits_per_record_avg
    hash_indexing_failures
    total_indexing_time
    indexing
    percent_indexed
    number_of_uses
    cleaning
    gc_stats
    cursor_stats
    dialect_stats
"""


def set_index_info(obj: Any, redis_client: Redis):
    index_info = redis_client.ft("paul_graham").info()
    for key in obj.__dataclass_fields__.keys():
        setattr(obj, key, index_info[key])
    return None


RedisIndexInfo = make_dataclass(
    "RedisIndexInfo",
    redis_index_info.split(),
    namespace={"__init__": set_index_info},
)


def redis_custom_schema():
    url = "redis://localhost:6379"
    redis_client = Redis.from_url(url)
    assert redis_client.ping() is True, "Cannot connect to Redis"
    schema = get_custom_schema()
    documents = SimpleDirectoryReader(get_data_dir()).load_data()
    # add updated_at metadata to documents
    modified_date: Any = None
    for document in documents:
        modified_date = document.metadata["last_modified_date"]
        document.metadata["updated_at"] = (  # type: ignore
            date_to_timestamp(modified_date)
        )

    if next_exists(redis_client.scan_iter("llama_index/*")):
        sys.exit("Index exists. Exiting")
    else:
        index = persist_redis_vector_store_index_schema(url, documents, schema)

    if modified_date:
        query = "What did the author learn?"
        retriever = create_retriever(index, date_to_timestamp(modified_date))
        print_query_nodes(retriever, query)

    print("----------redis keys-------------------")
    print_keys(redis_client)

    print("----------redis index attributes-------")
    rii = RedisIndexInfo(redis_client)
    print([a[1] for a in rii.attributes])

    doc_id = documents[0].doc_id
    print("Number of documents before deleting", redis_client.dbsize())
    index.vector_store.delete(doc_id)  # type: ignore
    print("Number of documents after deleting", redis_client.dbsize())
    index.vector_store.delete_index()  # type: ignore

    # on the redis client get the index with the command: FT.INFO paul_graham


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "default-vector-store": default_vector_store,
            "redis-vector-store": redis_vector_store,
            "redis-custom-schema": redis_custom_schema,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
