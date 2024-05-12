"""
https://cookbook.openai.com/examples/vector_databases/redis/using_redis_for_embeddings_search
"""

import logging
import pathlib
import tempfile
import urllib.request
import zipfile
from ast import literal_eval
from typing import Any

import numpy as np
import pandas as pd
import redis
from joblib import Memory
from openai import OpenAI
# import requests
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from redis_util import index_exists, print_indexing_failures

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


memory = Memory(tempfile.gettempdir(), verbose=0)


def get_wikipedia_embeddings_url():
    url = (
        "https://cdn.openai.com/API/examples/data"
        "/vector_database_wikipedia_articles_embedded.zip"
    )
    return url


def download_wikipedia_embeddings_zip_dataset(url):
    csv_data_file = "vector_database_wikipedia_articles_embedded.csv"
    df = None
    with urllib.request.urlopen(url) as f:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # download the zip wikipedia data into a temporary file
            temp_file.write(f.read())
            temp_file.seek(0)
            with zipfile.ZipFile(temp_file) as data_zip:
                # open a specified csv file from the zip file
                with data_zip.open(csv_data_file) as csv_data:
                    log.debug("about to read csv data")
                    df = pd.read_csv(csv_data)
                    log.debug("completed reading csv data")
                    df["title_vector"] = df.title_vector.apply(literal_eval)
                    log.debug("converted title_vector")
                    df["content_vector"] = df.content_vector.apply(
                        literal_eval
                    )
                    log.debug("converted content_vector")
            log.debug("Temporary file name: %s", temp_file.name)
    return df


def get_bike_model(redis_key: str, client):
    return client.json().get(redis_key, "$.model")


def get_bikes_descriptions(redis_key, client):
    keys: Any = sorted(client.keys(redis_key))
    descriptions = client.json().mget(keys, "$.description")
    return keys, [description[0] for description in descriptions]


def get_openai_embeddings(text: str):
    return (
        OpenAI()
        .embeddings.create(input=text, model=EMBEDDING_MODEL)
        .data[0]
        .embedding
    )


def save_embeddings(keys: list[str], embeddings: list[Any], client):
    pipeline = client.pipeline()
    for key, embedding in zip(keys, embeddings):
        pipeline.json().set(key, "$.description_embeddings", embedding)
    return pipeline.execute()


def get_query_text_list():
    queries = []
    return queries


def run_query(index_key, query, encoded_query, client: redis.Redis) -> Any:
    result = client.ft(index_key).search(
        query,
        {"query_vector": np.array(encoded_query, dtype=np.float32).tobytes()},
    )
    return result


def get_query_results(result_docs, query_text):
    results_list = []
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        results_list.append(
            {
                "query": query_text,
                "score": vector_score,
                "id": doc.id,
                "brand": doc.brand,
                "model": doc.model,
                "description": doc.description,
            }
        )
    return results_list


@memory.cache
def get_wikipedia_embeddings_dataframe(url) -> pd.DataFrame:
    df = download_wikipedia_embeddings_zip_dataset(url)
    # use memory.clear() to empty cache
    return df


def create_vector_index(
    index_key: str, dim: int, capacity: int, prefix: str, client: redis.Redis
):
    schema = (
        TextField(name="title"),
        TextField(name="url"),
        TextField(name="text"),
        VectorField(
            "title_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": capacity,
            },
        ),
        VectorField(
            "content_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": capacity,
            },
        ),
    )
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    res = client.ft(index_key).create_index(
        fields=schema, definition=definition
    )
    return res


def index_documents(prefix: str, df: pd.DataFrame, client: redis.Redis):
    for idx, srs in df.iterrows():
        key = f"{prefix}:{srs['id']}"

        title_embedding = np.array(
            srs["title_vector"], dtype=np.float32
        ).tobytes()
        content_embedding = np.array(
            srs["content_vector"], dtype=np.float32
        ).tobytes()

        srs["title_vector"] = title_embedding
        srs["content_vector"] = content_embedding

        client.hset(key, mapping=srs.to_dict())
        count: int = idx + 1  # type: ignore
        if count % 5000 == 0:
            print(f"{count} documents loaded")


def create_query(vector_field, return_fields, k, hybrid_fields):
    base_query = (
        f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
    )
    query = (
        Query(base_query)
        .sort_by("vector_score")
        .return_fields(*return_fields)
        .paging(0, k)
        .dialect(2)
    )
    return query


def search_redis(
    query_embeddings: list, index_key: str, query: Any, client: redis.Redis
):
    params_dict: Any = {
        "vector": np.array(query_embeddings).astype(dtype=np.float32).tobytes()
    }

    # perform vector search
    results = client.ft(index_key).search(query, params_dict)
    for i, article in enumerate(results.docs):  # type: ignore
        score = 1 - float(article.vector_score)
        print(f"{i}. {article.title} (Score: {round(score ,3) })")
    return results.docs  # type: ignore


def main():
    """
    72s to download, extract, write and read
    45s to download and extract
    over 5 minutes to convert embeddings from text format to numbers
    1.6s from cached
    """
    url = get_wikipedia_embeddings_url()
    df: pd.DataFrame = get_wikipedia_embeddings_dataframe(url)  # type: ignore
    print(df.shape)
    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")
    index_key = "embeddings-index"
    prefix = "doc"
    dim = len(df["title_vector"][0])
    capacity = len(df)

    if not index_exists(index_key, client):
        res = create_vector_index(index_key, dim, capacity, prefix, client)
        assert res == "OK", "Cannot create vector index"
        print_indexing_failures(index_key, client)

    index_documents(prefix, df, client)

    vector_field = "title_vector"
    return_fields = ["title", "url", "text", "vector_score"]
    hybrid_fields = "*"
    k = 5

    query = create_query(vector_field, return_fields, k, hybrid_fields)
    query_text = "modern art in Europe"
    embeddings = get_openai_embeddings(query_text)
    _ = search_redis(embeddings, index_key, query, client)
    print("Results are NOT CORRECT possibly because of incorrect embeddings")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    main()
