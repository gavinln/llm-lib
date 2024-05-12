import redis


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
