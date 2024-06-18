"""
https://haystack.deepset.ai/tutorials/34_extractive_qa_pipeline
"""

import logging
import pathlib
import sys

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_seven_wonders():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    return dataset


def get_qa_model():
    model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    return model


def get_doc_from_dataset(dataset) -> list[Document]:
    return [
        Document(content=doc["content"], meta=doc["meta"]) for doc in dataset
    ]


def main():
    dataset = get_seven_wonders()
    docs = get_doc_from_dataset(dataset)
    print(f"There are {len(docs)} documents")

    embedder = SentenceTransformersDocumentEmbedder(model=get_qa_model())
    document_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=embedder, name="embedder")
    indexing_pipeline.add_component(instance=writer, name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    indexing_pipeline.run({"documents": docs})

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()

    extractive_qa_pipeline = Pipeline()

    extractive_qa_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model=get_qa_model()),
        name="embedder",
    )
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")

    extractive_qa_pipeline.connect(
        "embedder.embedding", "retriever.query_embedding"
    )
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

    query = "Who was Pliny the Elder?"
    response = extractive_qa_pipeline.run(
        data={
            "embedder": {"text": query},
            "retriever": {"top_k": 3},
            "reader": {"query": query, "top_k": 2},
        }
    )
    for answer in response["reader"]["answers"]:
        print(f"{answer.query=} {answer.score=} {answer.data=}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
