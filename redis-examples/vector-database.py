"""
https://redis.io/docs/latest/develop/get-started/vector-database/
"""

import json
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (NumericField, TagField, TextField,
                                         VectorField)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from redis_util import index_exists, print_indexing_failures

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def get_bikes_dataset():
    url = (
        "https://raw.githubusercontent.com/bsbodden/"
        "redis_vss_getting_started/main/data/bikes.json"
    )
    response = requests.get(url)
    bikes = response.json()
    log.debug(json.dumps(bikes, indent=2))
    return bikes


def load_bikes(bikes, client: redis.Redis) -> list:
    pipeline = client.pipeline()
    for i, bike in enumerate(bikes, start=1):
        redis_key = f"bikes:{i:03}"
        log.debug(redis_key)
        pipeline.json().set(redis_key, "$", bike)
    res = pipeline.execute()
    return res


def get_bike_model(redis_key: str, client):
    return client.json().get(redis_key, "$.model")


def get_bikes_descriptions(redis_key, client):
    keys: Any = sorted(client.keys(redis_key))
    descriptions = client.json().mget(keys, "$.description")
    return keys, [description[0] for description in descriptions]


def get_embedder():
    embedder = SentenceTransformer("msmarco-distilbert-base-v4")
    return embedder


def get_embeddings(items: list[str]) -> list[list[np.float32]]:
    embedder = get_embedder()
    embeddings = (
        embedder.encode(items).astype(np.float32).tolist()  # type: ignore
    )
    return embeddings


def save_embeddings(keys: list[str], embeddings: list[Any], client):
    pipeline = client.pipeline()
    for key, embedding in zip(keys, embeddings):
        pipeline.json().set(key, "$.description_embeddings", embedding)
    return pipeline.execute()


def create_vector_index(index_key: str, dim: int, client: redis.Redis):
    schema = (
        TextField("$.model", no_stem=True, as_name="model"),
        TextField("$.brand", no_stem=True, as_name="brand"),
        NumericField("$.price", as_name="price"),
        TagField("$.type", as_name="type"),
        TextField("$.description", as_name="description"),
        VectorField(
            "$.description_embeddings",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )
    definition = IndexDefinition(prefix=["bikes:"], index_type=IndexType.JSON)
    res = client.ft(index_key).create_index(
        fields=schema, definition=definition
    )
    return res


def get_query_text_list():
    queries = [
        "Bike for small kids",
        "Best Mountain bikes for kids",
        "Cheap Mountain bike for kids",
        "Female specific mountain bike",
        "Road bike for beginners",
        "Commuter bike for people over 60",
        "Comfortable commuter bike",
        "Good bike for college students",
        "Mountain bike for beginners",
        "Vintage bike",
        "Comfortable city bike",
    ]
    return queries


def create_knn_query():
    query = (
        Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "id", "brand", "model", "description")
        .dialect(2)
    )
    return query


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


def main():
    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    bikes = get_bikes_dataset()
    print(json.dumps(bikes[2], indent=2))
    res = load_bikes(bikes, client)
    assert all(res), "Could not load bikes"

    res = get_bike_model("bikes:010", client)
    log.debug(print(res))

    keys, descriptions = get_bikes_descriptions("bikes:*", client)
    embeddings = get_embeddings(descriptions)
    VECTOR_DIMENSION = len(embeddings[0])
    log.info(f"{VECTOR_DIMENSION=}")
    res = save_embeddings(keys, embeddings, client)
    assert all(res), "Could not save embeddings"

    index_key = "idx:bikes_vss"
    if not index_exists(index_key, client):
        res = create_vector_index(index_key, VECTOR_DIMENSION, client)
        assert all(res), "Could not create vector index"
        print_indexing_failures(index_key, client)

    query_text_list = get_query_text_list()
    encoded_queries = get_embedder().encode(query_text_list)

    query = create_knn_query()

    # run all queries
    results_list = []
    for query_text, encoded_query in zip(query_text_list, encoded_queries):
        result = run_query(index_key, query, encoded_query, client)
        query_result = get_query_results(result.docs, query_text)
        results_list.extend(query_result)

    # convert results into dataframe and sort
    df = pd.DataFrame.from_records(results_list)
    df2 = df.sort_values(by=["query", "score"], ascending=[True, False])
    df2["description"] = df2["description"].apply(
        lambda x: (x[:497] + "...") if len(x) > 500 else x
    )
    print(df2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    main()
