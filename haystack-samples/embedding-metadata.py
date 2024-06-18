"""
https://haystack.deepset.ai/tutorials/39_embedding_metadata_for_improved_retrieval
"""

import logging
import pathlib
import sys
from typing import Sequence

import wikipedia
from haystack import Document, Pipeline

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from wikipedia.wikipedia import WikipediaPage

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_wiki_page_doc(page: WikipediaPage) -> Document:
    return Document(
        content=page.content, meta={"title": page.title, "url": page.url}
    )


def get_wikipedia_docs(titles: Sequence[str]) -> list[Document]:
    pages = [
        get_wiki_page_doc(wikipedia.page(title=title, auto_suggest=False))
        for title in titles
    ]
    return pages


def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large",
        meta_fields_to_embed=metadata_fields_to_embed,
    )
    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline


def main():
    bands = "The Beatles,The Cure".split(",")
    print(bands)

    docs = get_wikipedia_docs(bands)
    print(docs)

    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )
    document_store_with_embedded_metadata = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )
    indexing_pipeline = create_indexing_pipeline(document_store=document_store)
    indexing_with_metadata_pipeline = create_indexing_pipeline(
        document_store=document_store_with_embedded_metadata,
        metadata_fields_to_embed=["title"],
    )
    indexing_pipeline.run({"cleaner": {"documents": docs}})
    indexing_with_metadata_pipeline.run({"cleaner": {"documents": docs}})

    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="thenlper/gte-large"),
    )
    retrieval_pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store, scale_score=False, top_k=3
        ),
    )
    retrieval_pipeline.add_component(
        "retriever_with_embeddings",
        InMemoryEmbeddingRetriever(
            document_store=document_store_with_embedded_metadata,
            scale_score=False,
            top_k=3,
        ),
    )
    retrieval_pipeline.connect("text_embedder", "retriever")
    retrieval_pipeline.connect("text_embedder", "retriever_with_embeddings")

    result = retrieval_pipeline.run(
        {"text_embedder": {"text": "Have the Beatles ever been to Bangor?"}}
    )
    print("-------Retriever Results------------------")
    for doc in result["retriever"]["documents"]:
        print(doc.content)
    print("-------Retriever with Embeddings Results--")
    for doc in result["retriever_with_embeddings"]["documents"]:
        print(doc.content)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
