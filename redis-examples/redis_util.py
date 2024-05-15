import redis
from openai import OpenAI


def index_exists(index_key: str, client: redis.Redis):
    try:
        search = client.ft(index_key)
        search.info()
    except redis.ResponseError:
        return False
    return True


def print_indexing_failures(index_key: str, client):
    info = client.ft(index_key).info()
    num_docs = info["num_docs"]
    indexing_failures = info["hash_indexing_failures"]
    print(f"{num_docs} documents indexed with {indexing_failures} failures")


def get_embeddings(text: str, model="text-embedding-3-small") -> list[float]:
    """get openai text embedding for a single string"""
    text = text.replace("\n", " ")
    return (
        OpenAI().embeddings.create(input=[text], model=model).data[0].embedding
    )


def get_embeddings_batch(
    list_text: list[str] | tuple[str], model="text-embedding-3-small"
) -> list[list[float]]:
    """get openai text embedding for a list of strings"""
    list_text = [text.replace("\n", " ") for text in list_text]
    data = OpenAI().embeddings.create(input=list_text, model=model).data
    return [d.embedding for d in data]
