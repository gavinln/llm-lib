"""
https://github.com/redis/redis-vl-python
https://www.redisvl.com/user_guide/hybrid_queries_02.html
"""

import logging
import pathlib
import pickle
import sys
from typing import Any
from contextlib import contextmanager

import fire
import pandas as pd
import redis
from redisvl.index import SearchIndex
from redisvl.query import CountQuery, FilterQuery, RangeQuery, VectorQuery
from redisvl.query.filter import FilterExpression, Num, Tag, Text
from redisvl.schema import IndexSchema

# from redis.commands.search.field import TextField, VectorField
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType
# from redis.commands.search.query import Query


# from redis_util import get_embeddings, index_exists, print_indexing_failures


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def create_schema(schema_file: pathlib.Path) -> IndexSchema:
    if not schema_file.exists():
        print(f"Cannot find schema file {schema_file}")
        sys.exit()

    schema = IndexSchema.from_yaml(str(schema_file))
    return schema


def load_and_fetch_data(
    index: SearchIndex, list_data: list[Any], id_field: str, id_value
):
    # load list of dictionaries, specify the "id" field
    keys = index.load(list_data, id_field=id_field)
    print(keys)

    # fetch by "id"
    obj = index.fetch(id_value)
    return obj


def get_hybrid_example_data_schema():
    schema = {
        "index": {
            "name": "user_queries",
            "prefix": "user_queries_docs",
            "storage_type": "hash",  # default setting -- HASH
        },
        "fields": [
            {"name": "user", "type": "tag"},
            {"name": "credit_score", "type": "tag"},
            {"name": "job", "type": "text"},
            {"name": "age", "type": "numeric"},
            {"name": "office_location", "type": "geo"},
            {
                "name": "user_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }
    return schema


def get_hybrid_example_data_file():
    data_file = pathlib.Path(SCRIPT_DIR / "data" / "hybrid_example_data.pkl")
    assert data_file.exists(), f"Data file {data_file} does not exists"
    return data_file


def get_hybrid_example_data():
    data_file = get_hybrid_example_data_file()
    data = pickle.load(data_file.open("rb"))
    return data


def basic():
    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")
    schema_file = SCRIPT_DIR / "schemas" / "schema.yaml"

    schema: IndexSchema = create_schema(schema_file)

    # create search index
    index = SearchIndex(schema, client)
    index.create(overwrite=True)

    data = {
        "user": "john",
        "credit_score": "high",
        "embedding": [0.23, 0.49, -0.18, 0.95],
    }
    obj = load_and_fetch_data(index, [data], "user", "john")
    print(f"{obj=}")

    query = VectorQuery(
        # vector=[0.16, -0.34, 0.98, 0.23],
        vector=[0.23, 0.49, -0.18, 0.95],
        vector_field_name="embedding",
        num_results=3,
    )
    results = index.query(query)
    print(results)


@contextmanager
def create_index_and_load_data():
    schema = get_hybrid_example_data_schema()
    index_name = schema["index"]["name"]
    index = SearchIndex.from_dict(schema)
    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    index.set_client(client)

    # delete index if it exists
    if index_name in index.listall():
        index.delete()
    try:
        index.create(overwrite=True)

        data = get_hybrid_example_data()
        keys = index.load(data)
        assert len(keys) == len(data), "data cannot be loaded"
        yield index
    finally:
        index.delete()


def print_columns_except_id(results: list[dict]):
    "print all columns except id"
    df = pd.DataFrame(results)
    cols = [col for col in df.columns if not col == "id"]
    assert len(cols) > 0, "Only 1 column found"
    print(df[cols])


def get_vector_query():
    return VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "office_location",
        ],
    )


def vector_query():
    v = get_vector_query()
    with create_index_and_load_data() as index:
        results = index.query(v)
        print_columns_except_id(results)


def query_index_with_filter(filter_exp: FilterExpression) -> Any:
    v = get_vector_query()
    v.set_filter(filter_exp)
    with create_index_and_load_data() as index:
        results = index.query(v)
        return results


def tag_filters():
    t = Tag("credit_score") == "high"
    results = query_index_with_filter(t)
    print_columns_except_id(results)


def numeric_filters():
    t = Num("age") > 15
    results = query_index_with_filter(t)
    print_columns_except_id(results)


def text_filters():
    t = Text("job") == "doctor"
    results = query_index_with_filter(t)
    print_columns_except_id(results)


def combined_filters():
    t = Tag("credit_score") == "high"
    low = Num("age") >= 18
    high = Num("age") <= 100

    combined = t & low & high
    results = query_index_with_filter(combined)
    print_columns_except_id(results)


def filter_queries():
    has_low_credit = Tag("credit_score") == "low"
    filter_query = FilterQuery(
        return_fields=["user", "credit_score", "age", "job", "location"],
        filter_expression=has_low_credit,
    )
    with create_index_and_load_data() as index:
        results = index.query(filter_query)
        print_columns_except_id(results)


def count_queries():
    has_low_credit = Tag("credit_score") == "low"
    count_query = CountQuery(filter_expression=has_low_credit)
    with create_index_and_load_data() as index:
        count = index.query(count_query)
        print(f"{count} records match the filter expression {has_low_credit}")


def range_queries():
    range_query = RangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        distance_threshold=0.2,
    )
    with create_index_and_load_data() as index:
        results = index.query(range_query)
        print_columns_except_id(results)


def main():
    fire.Fire(
        {
            "redisvl-basic": basic,
            "redisvl-vector-query": vector_query,
            "redisvl-tag-filters": tag_filters,
            "redisvl-numeric-filters": numeric_filters,
            "redisvl-text-filters": text_filters,
            "redisvl-combined-filters": combined_filters,
            "redisvl-filter-queries": filter_queries,
            "redisvl-count-queries": count_queries,
            "redisvl-range-queries": range_queries,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
