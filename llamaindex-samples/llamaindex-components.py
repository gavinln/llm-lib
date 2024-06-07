"""
https://docs.llamaindex.ai/en/latest/understanding/loading/loading/
"""

import inspect
import logging
import pathlib
import sys

import fire

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser.interface import TextSplitter

# from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import (
    # DocstoreStrategy,
    # IngestionCache,
    IngestionPipeline,
)

# from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode

# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.vector_stores.types import BasePydanticVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data"))


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def load_directory_documents(data_dir) -> list[Document]:
    documents = SimpleDirectoryReader(
        data_dir, filename_as_id=True, recursive=False
    ).load_data()
    return documents


def get_text_splitter() -> TextSplitter:
    splitter = TokenTextSplitter(chunk_size=30, chunk_overlap=0)
    return splitter


def get_nodes_from_documents(documents) -> list[BaseNode]:
    "transform documents explicitly"
    splitter = get_text_splitter()
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def loading():
    print(inspect.currentframe().f_code.co_name)  # type: ignore
    data_dir = get_data_dir()
    documents = load_directory_documents(data_dir)
    print(f"There are {len(documents)} documents")
    for document in documents:
        print(document.text)
    nodes = get_nodes_from_documents(documents)
    print(f"There are {len(nodes)} nodes")
    for node in nodes:
        print(node.text)  # type: ignore
    splitter = get_text_splitter()
    pipeline = IngestionPipeline(transformations=[splitter])
    nodes = pipeline.run(documents=documents)
    print(f"There are {len(nodes)} nodes")
    for node in nodes:
        print(node.text)  # type: ignore
    breakpoint()


def main():
    fire.Fire(
        {
            "component-loading": loading,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
