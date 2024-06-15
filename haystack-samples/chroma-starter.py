"""
https://haystack.deepset.ai/overview/quick-start

OPENAI_API_KEY needed
"""

import logging
import os
import pathlib
import sys

import chromadb
import fire
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from chromadb.utils import embedding_functions

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def chroma_info():
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    print(client.get_version())
    print(client.get_settings().schema().keys())
    print(f"{client.heartbeat()=}")


def get_openai_key() -> str:
    key_name = "OPENAI_API_KEY"
    openai_key = os.environ.get(key_name, "")
    if len(openai_key) == 0:
        sys.exit(f"{key_name} is not set")
    return openai_key


def get_embedding_function() -> EmbeddingFunction:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=get_openai_key(), model_name="text-embedding-3-small"
    )
    return openai_ef


def add_collection(client: ClientAPI, ef: EmbeddingFunction) -> Collection:
    collection = client.create_collection(
        name="my_collection", embedding_function=ef
    )
    collection.add(
        documents=[
            "This farm grows pineapples",
            "This orchard grows oranges",
        ],
        metadatas=[
            {"land_type": "farm", "plant_height": 3},
            {"land_type": "orchard", "plant_height": 12},
        ],
        ids=["id1", "id2"],
    )
    return collection


def chroma_collections():
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    print(f"{client.count_collections()=}")
    print(f"{client.list_collections()=}")
    ef = get_embedding_function()

    print("-----add collection------")
    collection = add_collection(client, ef)
    print(f"{client.count_collections()=}")
    print(f"{collection.count()=}")

    query1 = "This is a query document about Hawaii"
    results1 = collection.query(query_texts=[query1], n_results=2)
    print(f"{query1=}")
    print(results1)

    query2 = "This is a query document about Florida"
    results2 = collection.query(query_texts=[query2], n_results=2)
    print(f"{query2=}")
    print(results2)

    print("-----delete document------")
    collection.delete(ids=["id2"])
    print(f"{collection.count()=}")

    print("-----delete document------")
    collection.delete(ids=["id1"])
    print(f"{collection.count()=}")

    client.delete_collection(name="my_collection")
    print(f"{client.count_collections()=}")


def chroma_query():
    client = chromadb.Client(
        Settings(anonymized_telemetry=False, allow_reset=True)
    )
    ef = get_embedding_function()

    collection = add_collection(client, ef)

    results1 = collection.get(ids=["id2"], include=["documents"])
    print(results1)

    results2 = collection.get(where={"land_type": "farm"})
    print(results2)

    results3 = collection.get(where={"plant_height": {"$gt": 6}})
    print(results3)

    results4 = collection.get(where_document={"$contains": "farm"})
    print(results4)

    # client.delete_collection(name="my_collection")
    client.reset()
    print(f"{client.count_collections()=}")


def main():
    fire.Fire(
        {
            "chroma-info": chroma_info,
            "chroma-collections": chroma_collections,
            "chroma-query": chroma_query,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
