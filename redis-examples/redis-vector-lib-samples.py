"""
https://github.com/redis/redis-vl-python
"""

import logging
import pathlib
import sys

import redis

# from redis.commands.search.field import TextField, VectorField
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType
# from redis.commands.search.query import Query

from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex

# from redis_util import get_embeddings, index_exists, print_indexing_failures


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")
    schema_file = SCRIPT_DIR / "schemas" / "schema.yaml"
    if not schema_file.exists():
        print(f"Cannot find schema file {schema_file}")
        sys.exit()

    schema = IndexSchema.from_yaml(str(schema_file))
    index = SearchIndex(schema, client)
    index.create()

    data = {
        "user": "john",
        "credit_score": "high",
        "embedding": [0.23, 0.49, -0.18, 0.95],
    }
    # load list of dictionaries, specify the "id" field
    index.load([data], id_field="user")

    # fetch by "id"
    obj = index.fetch("john")
    print(f"{obj=}")
    # breakpoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
