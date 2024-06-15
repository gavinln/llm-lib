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
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers.document_writer import DocumentWriter
from haystack.core.component.component import Component
from haystack_integrations.document_stores.chroma.document_store import (
    ChromaDocumentStore,
)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_doc_path() -> str:
    return str(SCRIPT_DIR / "data" / "davinci.txt")


def get_component_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.to_dict()["components"].keys())


def get_persist_path(writer: DocumentWriter) -> str:
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


def remove_uuid_dirs(base_path: pathlib.Path):
    paths = base_path.glob("*")
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
    persist_path = pathlib.Path(get_persist_path(writer))
    chromadb_file = persist_path / "chroma.sqlite3"
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
    remove_uuid_dirs(persist_path)


def rag_pipeline():
    doc_path = get_doc_path()
    breakpoint()


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
