"""
https://redis.io/docs/latest/develop/get-started/data-store/
"""

import logging
import pathlib

import redis

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def redis_strings(client):
    """save and load strings to redis

    Args:
        client: redis client
    """
    res = client.set("bike:1", "Process 134")
    print(res)

    res = client.get("bike:1")
    print(res)


def redis_hashes(client):
    """save and load hashes to redis

    Args:
        client: redis client
    """
    res1 = client.hset(
        "cycle:1",
        mapping={
            "model": "Deimos",
            "brand": "Ergonom",
            "type": "Enduro bikes",
            "price": 4972,
        },
    )
    print(res1)

    res2 = client.hget("cycle:1", "model")
    print(res2)

    res3 = client.hget("cycle:1", "price")
    print(res3)

    res4 = client.hgetall("cycle:1")
    print(res4)


def main():
    r = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True,
    )
    redis_strings(r)
    redis_hashes(r)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
