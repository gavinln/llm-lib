"""
https://redis.io/docs/latest/develop/get-started/document-database/
"""

import json
import logging
import pathlib
from typing import Any

import redis
from redis.commands.json.path import Path
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
# import redis.commands.search.aggregation as aggregations
# import redis.commands.search.reducers as reducers
# from redis.commands.search.query import NumericFilter
from redis.commands.search.query import Query

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def get_bicycles():
    bicycles = [
        {
            "brand": "Velorim",
            "model": "Jigger",
            "description": "Velorim Jigger",
            "price": 270,
            "condition": "new",
        },
        {
            "brand": "Noka Bikes",
            "model": "Kahuna",
            "description": "Noka Bikes Kahuna",
            "price": 3200,
            "condition": "used",
        },
        {
            "brand": "Ergonom",
            "model": "Deimos",
            "description": "Ergonom Deimos",
            "price": 4972,
            "condition": "new",
        },
    ]
    return bicycles


def get_bicycle_schema():
    schema = (
        TextField("$.brand", as_name="brand"),
        TextField("$.model", as_name="model"),
        TextField("$.description", as_name="description"),
        NumericField("$.price", as_name="price"),
        TagField("$.condition", as_name="condition"),
    )
    return schema


def main():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    print(r.exists("idx:bicycle"))
    rs = r.ft("idx:bicycle")
    schema = get_bicycle_schema()
    rs.create_index(
        schema,
        definition=IndexDefinition(
            prefix=["bicycle:"], index_type=IndexType.JSON
        ),
    )
    # store json objects
    for bid, bicycle in enumerate(get_bicycles()):
        r.json().set(f"bicycle:{bid}", Path.root_path(), bicycle)

    # search for documents
    res: Any = rs.search(Query("*"))
    print("Documents found:", res.total)

    # find specific bicycle
    res = rs.search(Query("@model:Jigger"))
    print(res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
