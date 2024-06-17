"""
https://haystack.deepset.ai/tutorials/31_metadata_filtering
"""

import logging
import pathlib
import sys
from datetime import datetime

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_documents() -> list[Document]:
    documents = [
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]
    return documents


def print_retrieved_documents(response):
    retrieved_docs = response["retriever"]["documents"]
    print(f"There are {len(retrieved_docs)} documents retrieved")
    for doc in retrieved_docs:
        print(doc.meta)


def main():
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=get_documents())
    print(f"There are {document_store.count_documents()} documents")

    retriever = InMemoryBM25Retriever(document_store=document_store)

    pipeline = Pipeline()
    pipeline.add_component(instance=retriever, name="retriever")

    query = "Haystack installation"
    response = pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {
                    "field": "meta.version",
                    "operator": ">",
                    "value": 1.21,
                },
            }
        }
    )
    print_retrieved_documents(response)
    response = pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "meta.version",
                            "operator": ">",
                            "value": 1.21,
                        },
                        {
                            "field": "meta.date",
                            "operator": ">",
                            "value": datetime(2023, 11, 7),
                        },
                    ],
                },
            }
        }
    )
    print_retrieved_documents(response)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
