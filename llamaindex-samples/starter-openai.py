"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib
import sys

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def main():
    documents = SimpleDirectoryReader("data").load_data()
    index: BaseIndex = VectorStoreIndex.from_documents(
        documents,
    )
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    main()
