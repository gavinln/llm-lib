"""
https://cookbook.openai.com/examples/vector_databases/redis/redisqna/redisqna
"""

import logging
import pathlib
import tempfile
import zipfile
from typing import Any, Generator, NamedTuple

import numpy as np
import redis
from joblib import Memory
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


def get_openai_embeddings(text: str):
    return np.array(
        OpenAI()
        .embeddings.create(input=text, model=EMBEDDING_MODEL)
        .data[0]
        .embedding,
        dtype=np.float32,
    )


def create_vector_index(
    index_name: str, dim: int, prefix: str, client: redis.Redis
):
    schema = (
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
        TextField("$.content", as_name="content"),
    )
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
    res = client.ft(index_name).create_index(
        fields=schema, definition=definition
    )
    return res


def create_query(vector_field, return_field) -> Query:
    base_query = f"*=>[KNN 1 @{vector_field} $query_vec AS vector_score]"
    query = (
        Query(base_query)
        .sort_by("vector_score")
        .return_fields(return_field)
        .dialect(2)
    )
    return query


def search_redis(
    query_vec: np.ndarray, index_name: str, query: Any, client: redis.Redis
):
    params: Any = {"query_vec": query_vec.tobytes()}
    result: Any = client.ft(index_name).search(query, query_params=params)
    assert len(result.docs) > 0
    return result.docs[0].content


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


class TextEmbedding(NamedTuple):
    text: bytes
    embedding: np.ndarray

    def __repr__(self):
        return "text len: {}, embedding len: {}".format(
            len(self.text), len(self.embedding)
        )


def load_text_embeddings(
    text_embeddings: list[TextEmbedding], client: redis.Redis
) -> list[bool]:

    res_list = []
    for i, text_embedding in enumerate(text_embeddings):
        res = client.json().set(
            f"doc:{i + 1}",
            "$",
            {
                "content": text_embedding.text.decode("utf-8"),
                "vector": text_embedding.embedding.tolist(),
            },
        )
        res_list.append(res)
    return res_list


def get_text_embeddings(text_zip_file: pathlib.Path) -> list[TextEmbedding]:
    text_embeddings: list[TextEmbedding] = []
    for article in get_articles(text_zip_file):
        embedding = get_openai_embeddings(article.decode("utf-8"))
        text_embedding = TextEmbedding(article, embedding)
        text_embeddings.append(text_embedding)
    return text_embeddings


def main():
    prompt = (
        "Is Sam Bankman-Fried's company, FTX,"
        " considered a well-managed company?"
    )
    response = get_completion(prompt)
    embeddings = get_openai_embeddings(response)
    print(response)
    print(len(embeddings))

    article_file = get_articles_file()
    text_embeddings: list[TextEmbedding] = get_text_embeddings(article_file)

    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")

    index_name = "idx"
    prefix = "doc:"
    assert len(text_embeddings) > 0, "There are not articles"
    dim = len(text_embeddings[0].text)

    if not index_exists(index_name, client):
        res = create_vector_index(index_name, dim, prefix, client)
        assert res == "OK", "Cannot create vector index"
        print_indexing_failures(index_name, client)

    res_list = load_text_embeddings(text_embeddings, client)
    assert all(res_list), "Not all text embeddings were loaded"

    prompt_embedding = get_openai_embeddings(prompt)
    vector_field = "vector"
    return_field = "content"
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
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    main()
