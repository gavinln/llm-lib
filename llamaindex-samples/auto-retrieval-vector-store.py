"""
https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_auto_retriever/

OPENAI_API_KEY needed

Many vector databases support metadata filters in addition to semantic
 search using a query string.

Auto retrieval uses a combination of metadata filters and semantic filters in
 combination or each one alone.
"""

import logging
import pathlib
from typing import Any

from llama_index.core import GPTVectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import (
    MetadataInfo,
    VectorStore,
    VectorStoreInfo,
)
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def print_query_response(query_engine, query):
    response = query_engine.query(query)
    print(f"--{query}----")
    print(response)


def print_query_nodes(retriever: BaseRetriever, query):
    result_nodes = retriever.retrieve(query)
    print(f"--{query}----")
    no_nodes = True
    for node in result_nodes:
        print(node)
        no_nodes = False
    if no_nodes:
        print("There are NO NODES")


def create_retriever(index: BaseIndex) -> BaseRetriever:
    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(key="tag", value="target"),
            ],
        ),
    )
    return retriever


def get_nodes() -> list[TextNode]:
    return [
        TextNode(
            text=(
                "Michael Jordan is a retired professional basketball player,"
                " widely regarded as one of the greatest basketball players of"
                " all time."
            ),
            metadata={  # type: ignore
                "category": "Sports",
                "country": "United States",
            },
        ),
        TextNode(
            text=(
                "Angelina Jolie is an American actress, filmmaker, and"
                " humanitarian. She has received numerous awards for her"
                "acting and is known for her philanthropic work."
            ),
            metadata={  # type: ignore
                "category": "Entertainment",
                "country": "United States",
            },
        ),
        TextNode(
            text=(
                "Elon Musk is a business magnate, industrial designer, and"
                " engineer. He is the founder, CEO, and lead designer of"
                " SpaceX, Tesla, Inc., Neuralink, and The Boring Company."
            ),
            metadata={  # type: ignore
                "category": "Business",
                "country": "United States",
            },
        ),
        TextNode(
            text=(
                "Rihanna is a Barbadian singer, actress, and businesswoman."
                " She has achieved significant success in the music industry"
                " and is known for her versatile musical style."
            ),
            metadata={  # type: ignore
                "category": "Music",
                "country": "Barbados",
            },
        ),
        TextNode(
            text=(
                "Cristiano Ronaldo is a Portuguese professional footballer"
                " who is considered one of the greatest football players"
                " of all time. Hehas won numerous awards and set multiple"
                " records during his career."
            ),
            metadata={  # type: ignore
                "category": "Sports",
                "country": "Portugal",
            },
        ),
    ]


def create_index_from_nodes(nodes, vector_store) -> BaseIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
    return index


def main():
    """
    DOES NOT WORK correctly for DocArrayInMemoryVectorStore
    """
    nodes = get_nodes()
    vector_store: VectorStore = DocArrayInMemoryVectorStore()
    index: Any = create_index_from_nodes(nodes, vector_store)
    vector_store_info = VectorStoreInfo(
        content_info="brief biography of celebrities",
        metadata_info=[
            MetadataInfo(
                name="category",
                type="str",
                description=(
                    "Category of the celebrity, one of [Sports, Entertainment,"
                    " Business, Music]"
                ),
            ),
            MetadataInfo(
                name="country",
                type="str",
                description=(
                    "Country of the celebrity, one of [United States,"
                    " Barbados, Portugal]"
                ),
            ),
        ],
    )
    retriever = VectorIndexAutoRetriever(
        index, vector_store_info=vector_store_info
    )

    print_query_nodes(
        retriever, "Tell me about two celebrities from United States"
    )
    print_query_nodes(
        retriever, "Tell me about Sports celebrities from United States"
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
