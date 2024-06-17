"""
https://haystack.deepset.ai/tutorials/30_file_type_preprocessing_index_pipeline
"""

import logging
import pathlib
import sys

from haystack import Pipeline
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_document_dir():
    return SCRIPT_DIR / "recipe_files"


def main():
    document_store = InMemoryDocumentStore()
    file_type_router = FileTypeRouter(
        mime_types=["text/plain", "application/pdf", "text/markdown"]
    )
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(
        split_by="word", split_length=150, split_overlap=50
    )

    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(
        instance=file_type_router, name="file_type_router"
    )
    preprocessing_pipeline.add_component(
        instance=text_file_converter, name="text_file_converter"
    )
    preprocessing_pipeline.add_component(
        instance=markdown_converter, name="markdown_converter"
    )
    preprocessing_pipeline.add_component(
        instance=pdf_converter, name="pypdf_converter"
    )
    preprocessing_pipeline.add_component(
        instance=document_joiner, name="document_joiner"
    )
    preprocessing_pipeline.add_component(
        instance=document_cleaner, name="document_cleaner"
    )
    preprocessing_pipeline.add_component(
        instance=document_splitter, name="document_splitter"
    )
    preprocessing_pipeline.add_component(
        instance=document_embedder, name="document_embedder"
    )
    preprocessing_pipeline.add_component(
        instance=document_writer, name="document_writer"
    )

    preprocessing_pipeline.connect(
        "file_type_router.text/plain", "text_file_converter.sources"
    )
    preprocessing_pipeline.connect(
        "file_type_router.application/pdf", "pypdf_converter.sources"
    )
    preprocessing_pipeline.connect(
        "file_type_router.text/markdown", "markdown_converter.sources"
    )
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    doc_dir = get_document_dir()
    files = list(doc_dir.glob("**/*"))
    print(files)

    result = preprocessing_pipeline.run(
        {"file_type_router": {"sources": files}}
    )
    print(result)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
