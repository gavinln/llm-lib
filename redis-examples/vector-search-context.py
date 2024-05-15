"""
https://cookbook.openai.com/examples/vector_databases/redis/redisqna/redisqna
"""

import logging
import pathlib
import zipfile
from typing import Any, Generator

import numpy as np
import redis
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from redis_util import index_exists, print_indexing_failures

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-3.5-turbo"


def get_articles_file() -> pathlib.Path:
    articles_file = SCRIPT_DIR / "assets" / "news-articles.zip"
    assert articles_file.exists(), f"Missing file {articles_file}"
    return articles_file


def get_articles(article_file: pathlib.Path) -> Generator[bytes, None, None]:
    """
    Extracts and yields the contents of each article in a given zip file.

    Parameters:
    article_file (pathlib.Path): The path to the zip file containing articles.

    Yields:
    bytes: The contents of each article in the zip file.
    """

    with zipfile.ZipFile(article_file) as article_zip:
        for name in article_zip.namelist():
            yield article_zip.open(name, "r").read()


def get_openai_embeddings(text: str) -> list[float]:
    return (
        OpenAI()
        .embeddings.create(input=text, model=EMBEDDING_MODEL)
        .data[0]
        .embedding
    )


def create_vector_index(
    index_name: str, dim: int, prefix: str, client: redis.Redis
):
    schema = (
        TextField("$.text", as_name="text"),
        VectorField(
            "$.vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
    res = client.ft(index_name).create_index(
        fields=schema, definition=definition
    )
    return res


def create_query(vector_field, return_field) -> Query:
    base_query = f"(*)=>[KNN 1 @{vector_field} $query_vec AS vector_score]"
    query = (
        Query(base_query)
        .sort_by("vector_score")
        .return_fields(return_field)
        .dialect(2)
    )
    return query


def search_redis(
    query_vec: list[float],
    index_name: str,
    query: Any,
    client: redis.Redis,
):
    query_vec_bytes = np.array(query_vec).astype(np.float32).tobytes()
    params: Any = {"query_vec": query_vec_bytes}
    result: Any = client.ft(index_name).search(query, query_params=params)
    assert len(result.docs) > 0
    return result.docs[0].text


def get_completion(prompt: str, model=OPENAI_MODEL) -> str:
    completion: ChatCompletion = OpenAI().chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0,
    )
    text = completion.choices[0].message.content
    assert text, "Cannot get completion"
    return text


def load_text_embeddings(
    articles: list[bytes], embeddings: list[list[float]], client: redis.Redis
) -> list[bool]:

    res_list = []
    for i, (article, embedding) in enumerate(zip(articles, embeddings)):
        key = f"doc:{i + 1}"
        res = client.json().set(
            key,
            "$",
            {
                "text": article.decode("utf-8"),
                "vector": np.array(embedding).astype(np.float32).tolist(),
            },
        )
        res_list.append(res)
    return res_list


def get_text_embeddings(articles: list[bytes]) -> list[list[float]]:
    return [
        get_openai_embeddings(article.decode("utf-8")) for article in articles
    ]


def main():
    prompt = (
        "Is Sam Bankman-Fried's company, FTX,"
        " considered a well-managed company?"
    )
    response = get_completion(prompt)
    print(response)

    article_file = get_articles_file()
    articles = list(get_articles(article_file))[:4]  # TODO
    embeddings: list[list[float]] = get_text_embeddings(articles)

    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")

    index_name = "idx"
    prefix = "doc:"
    assert len(embeddings) > 0, "There are no articles"
    dim = len(embeddings[0])

    if not index_exists(index_name, client):
        res = create_vector_index(index_name, dim, prefix, client)
        assert res == "OK", "Cannot create vector index"
        print_indexing_failures(index_name, client)

    res_list = load_text_embeddings(articles, embeddings, client)
    assert all(res_list), "Not all text embeddings were loaded"

    prompt_embedding = get_openai_embeddings(prompt)

    vector_field = "vector"
    return_field = "text"
    query = create_query(vector_field, return_field)
    context = search_redis(prompt_embedding, index_name, query, client)
    prompt_with_context = """

    Using the information delimited by triple backticks, answer this question:
    Is Sam Bankman-Fried's company, FTX, considered a well-managed company?

    Context: ```{}```
    """.format(
        context
    )
    new_response = get_completion(prompt_with_context)
    print(new_response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
