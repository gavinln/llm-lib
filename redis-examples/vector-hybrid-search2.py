"""
document
https://cookbook.openai.com/examples/vector_databases/redis/redis-hybrid-query-examples

notebook
https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/redis/redis-hybrid-query-examples.ipynb

python file
https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py

data
https://github.com/openai/openai-cookbook/blob/main/examples/data/styles_2k.csv
"""

import itertools
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import redis
from redis.commands.search.field import (NumericField, TagField, TextField,
                                         VectorField)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from redis_util import (get_embeddings, get_embeddings_batch, index_exists,
                        print_indexing_failures)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


def get_styles_file() -> pathlib.Path:
    file_path = SCRIPT_DIR / "data" / "styles_2k.csv.gz"
    assert file_path.exists(), f"Cannot find file {file_path}"
    return file_path


def get_styles_data(styles_file) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(styles_file, header=0, compression="gzip")
    df2 = df.dropna().rename({"id": "product_id"}, axis=1)
    df2["product_text"] = df2.apply(
        lambda row: """
            name {} category {} subcategory {} color {} gender {}
            """.format(
            row["productDisplayName"],
            row["masterCategory"],
            row["subCategory"],
            row["baseColour"],
            row["gender"],
        )
        .strip()
        .replace("\n", " ")
        .lower(),
        axis=1,
    )
    return df2


def create_vector_index(
    index_name: str, dim: int, prefix: str, client: redis.Redis
):
    """
    product_id, gender, masterCategory, subCategory, articleType,
           baseColour, season, year, usage, productDisplayName,
    """
    schema = (
        TextField(name="productDisplayName"),
        TagField(name="masterCategory"),
        TagField(name="articleType"),
        TagField(name="gender"),
        TagField(name="season"),
        NumericField(name="year"),
        TextField(name="text"),
        VectorField(
            "product_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    res = client.ft(index_name).create_index(
        fields=schema, definition=definition
    )
    return res


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def index_documents(prefix: str, df: pd.DataFrame, client: redis.Redis):
    batchsize = 500

    pipe = client.pipeline()
    for idx, (_, doc) in enumerate(df.iterrows()):
        key = f"{prefix}:{doc['product_id']}"
        text_embedding = np.array(
            doc["product_vector"], dtype=np.float32
        ).tobytes()
        doc["product_vector"] = text_embedding
        pipe.hset(key, mapping=doc.to_dict())
        if (idx + 1) % batchsize == 0:
            pipe.execute()
    pipe.execute()


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
    query_embeddings: list, index_name: str, query: Any, client: redis.Redis
):
    params_dict: Any = {
        "vector": np.array(query_embeddings).astype(dtype=np.float32).tobytes()
    }

    # perform vector search
    results = client.ft(index_name).search(query, params_dict)
    for i, product in enumerate(results.docs):  # type: ignore
        score = 1 - float(product.vector_score)
        print(f"{i}. {product.productDisplayName} (Score: {round(score ,3) })")
    return results.docs  # type: ignore


def query_redis(query_text, hybrid_fields, index_name, client):
    print(f"{query_text=} {hybrid_fields=}")

    vector_field = "product_vector"
    return_fields = [
        "productDisplayName",
        "masterCategory",
        "gender",
        "season",
        "year",
        "vector_score",
    ]

    k = 10
    query = create_query(vector_field, return_fields, k, hybrid_fields)
    query_embeddings = get_embeddings(query_text)
    search_redis(query_embeddings, index_name, query, client)


def main():
    styles_file = get_styles_file()
    df = get_styles_data(styles_file)
    print(f"Styles data {df.shape}")

    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")

    batchsize = 1000

    embeddings = [
        get_embeddings_batch(text)
        for text in batched(df.product_text, batchsize)
    ]
    df["product_vector"] = list(itertools.chain(*embeddings))

    index_name = "product_embeddings"
    prefix = "doc"

    assert df.product_vector.size > 0, "Cannot get embeddings"
    dim = len(df.product_vector[0])

    if not index_exists(index_name, client):
        res = create_vector_index(index_name, dim, prefix, client)
        assert res == "OK", "Cannot create vector index"
        print_indexing_failures(index_name, client)

    index_documents(prefix, df, client)

    query_text = "man blue jeans"
    hybrid_fields = "*"
    query_redis(query_text, hybrid_fields, index_name, client)

    query_text = "shirt"
    hybrid_fields = '@productDisplayName:"slim fit"'
    query_redis(query_text, hybrid_fields, index_name, client)

    query_text = "shirt"
    hybrid_fields = "@masterCategory:{Accessories}"
    query_redis(query_text, hybrid_fields, index_name, client)

    query_text = "sandals"
    hybrid_fields = "@year:[2011 2012]"
    query_redis(query_text, hybrid_fields, index_name, client)

    query_text = "blue sandals"
    hybrid_fields = "(@year:[2011 2012] @season:{Summer})"
    query_redis(query_text, hybrid_fields, index_name, client)

    query_text = "brown belt"
    hybrid_fields = (
        "(@year:[2012 2012] @articleType:{Shirts | Belts}"
        '@productDisplayName:"Wrangler")'
    )
    query_redis(query_text, hybrid_fields, index_name, client)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
