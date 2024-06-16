"""
https://haystack.deepset.ai/overview/quick-start

OPENAI_API_KEY needed
"""

import logging
import pathlib
import shutil
import sys
from uuid import UUID

import fire
from haystack import Pipeline, PredefinedPipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.core.component.component import Component
from haystack_integrations.components.retrievers.chroma import (
    ChromaEmbeddingRetriever,
)
from haystack_integrations.document_stores.chroma.document_store import (
    ChromaDocumentStore,
)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_doc_path() -> str:
    return str(SCRIPT_DIR / "data" / "davinci.txt")


def get_temp_storage_dir():
    path = pathlib.Path(SCRIPT_DIR / "temp_storage")
    if not path.exists():
        path.mkdir()
    return path


def get_component_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.to_dict()["components"].keys())


def get_persist_path(writer: DocumentWriter) -> str | None:
    assert isinstance(writer.document_store, ChromaDocumentStore)
    chroma_store: ChromaDocumentStore = writer.document_store
    return chroma_store._persist_path


def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if uuid_to_test is a valid UUID.
    https://stackoverflow.com/questions/53847404/how-to-check-uuid-validity-in-python

    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


def remove_uuid_dirs(base_path: str):
    paths = pathlib.Path(base_path).glob("*")
    for path in paths:
        dir_name = path.resolve().name
        if path.is_dir() and is_valid_uuid(dir_name):
            print(f"Recursively removing {path}")
            shutil.rmtree(path)


def remove_chromadb_file(pipeline: Pipeline):
    component_names = get_component_names(pipeline)
    assert "writer" in component_names
    writer: Component = pipeline.get_component("writer")
    assert isinstance(writer, DocumentWriter)
    persist_path = get_persist_path(writer)
    if persist_path:
        chromadb_file = pathlib.Path(persist_path) / "chroma.sqlite3"
        chromadb_file.unlink()
    return persist_path


def rag_predefined():
    doc_path = get_doc_path()

    indexing_pipeline = Pipeline.from_template(PredefinedPipeline.INDEXING)
    assert isinstance(indexing_pipeline, Pipeline)

    indexing_pipeline.run(data={"sources": [doc_path]})

    rag_pipeline = Pipeline.from_template(PredefinedPipeline.RAG)
    assert isinstance(rag_pipeline, Pipeline)

    query = "How old was he when he died?"

    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": query},
            "text_embedder": {"text": query},
        }
    )
    print(result["llm"]["replies"][0])

    # clean up implicitly used chromadb in current directory
    persist_path = remove_chromadb_file(indexing_pipeline)
    if persist_path:
        remove_uuid_dirs(persist_path)


def rag_pipeline():
    temp_storage = get_temp_storage_dir()

    text_file_converter = TextFileToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter()
    embedder = OpenAIDocumentEmbedder()
    document_store = ChromaDocumentStore(persist_path=str(temp_storage))
    writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", text_file_converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    doc_path = get_doc_path()
    indexing_pipeline.run(data={"sources": [doc_path]})

    text_embedder = OpenAITextEmbedder()
    retriever = ChromaEmbeddingRetriever(document_store)
    template = """Given these documents, answer the question.
                  Documents:
                  {% for doc in documents %}
                      {{ doc.content }}
                  {% endfor %}
                  Question: {{query}}
                  Answer:"""
    prompt_builder = PromptBuilder(template=template)
    llm = OpenAIGenerator()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect(
        "text_embedder.embedding", "retriever.query_embedding"
    )
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    query = "How old was he when he died?"
    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": query},
            "text_embedder": {"text": query},
        }
    )
    print(result["llm"]["replies"][0])

    shutil.rmtree(temp_storage)


def main():
    fire.Fire(
        {
            "rag-predefined": rag_predefined,
            "rag-pipeline": rag_pipeline,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
