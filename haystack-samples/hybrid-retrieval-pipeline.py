"""
https://haystack.deepset.ai/tutorials/33_hybrid_retrieval
"""

import logging
import pathlib
import sys
from typing import Any

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors.document_splitter import (
    DocumentSplitter,
)
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_pubmed_dataset():
    dataset: Any = load_dataset("anakin87/medrag-pubmed-chunk", split="train")
    return dataset.select(range(50))


def get_sentence_transformer_model_name():
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_pubmed_docs() -> list[Document]:
    dataset = get_pubmed_dataset()
    doc: Any
    docs: list[Document] = []
    for doc in dataset:
        docs.append(
            Document(
                content=doc["contents"],
                meta={
                    "title": doc["title"],
                    "abstract": doc["content"],
                    "pmid": doc["id"],
                },
            )
        )
    return docs


def pretty_print_results(prediction):
    for idx, doc in enumerate(prediction["documents"]):
        print(f"---------index {idx}------------")
        print(doc.meta["title"], "\t", doc.score)
        print(doc.meta["abstract"])


def main():
    docs: list[Document] = get_pubmed_docs()
    print(f"There are {len(docs)} pubmed documents")
    document_splitter = DocumentSplitter(
        split_by="word", split_length=512, split_overlap=32
    )
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=get_sentence_transformer_model_name()
    )
    document_store = InMemoryDocumentStore()
    document_writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("document_splitter", document_splitter)
    indexing_pipeline.add_component("document_embedder", document_embedder)
    indexing_pipeline.add_component("document_writer", document_writer)

    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    indexing_pipeline.run({"document_splitter": {"documents": docs}})

    text_embedder = SentenceTransformersTextEmbedder(
        model=get_sentence_transformer_model_name()
    )
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=3)
    bm25_retriever = InMemoryBM25Retriever(document_store, top_k=3)

    document_joiner = DocumentJoiner()

    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)

    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker")

    query = "hypoxia in infants"

    result = hybrid_retrieval.run(
        {
            "text_embedder": {"text": query},
            "bm25_retriever": {"query": query},
            "ranker": {"query": query},
        }
    )
    pretty_print_results(result["ranker"])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
