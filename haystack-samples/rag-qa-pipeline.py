"""
https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_seven_wonders():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    return dataset


def get_non_null_columns(dataset: Dataset) -> list[str]:
    df = dataset.to_pandas()
    cols = df.columns[df.isna().sum() != df.shape[0]].tolist()  # type: ignore
    return cols


def get_sentence_transformer_model():
    return "sentence-transformers/all-MiniLM-L6-v2"


def print_answer(pipeline, question):
    response = pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
        }
    )
    print(response["llm"]["replies"])


def main():
    # get documents
    dataset = get_seven_wonders()
    print(f"dataset columns = {dataset.column_names}")  # type: ignore
    docs = [
        Document(content=doc["content"], meta=doc["meta"])  # type: ignore
        for doc in dataset
    ]
    print(f"There are {len(docs)} documents")

    # get embeddings
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=get_sentence_transformer_model()
    )
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)

    # write documents & embeddings to memory store
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_with_embeddings["documents"])

    # get a text embedder
    text_embedder = SentenceTransformersTextEmbedder(
        model=get_sentence_transformer_model()
    )

    # get a retriever
    retriever = InMemoryEmbeddingRetriever(document_store)

    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    # create a prompt builder
    prompt_builder = PromptBuilder(template=template)

    # create a generator
    generator = OpenAIGenerator(model="gpt-3.5-turbo")

    # build the pipeline
    basic_rag_pipeline = Pipeline()

    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect(
        "text_embedder.embedding", "retriever.query_embedding"
    )
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    example_questions = [
        "Where is Gardens of Babylon?",
        "Why did people build Great Pyramid of Giza?",
        "What does Rhodes Statue look like?",
        "Why did people visit the Temple of Artemis?",
        "What is the importance of Colossus of Rhodes?",
        "What happened to the Tomb of Mausolus?",
        "How did Colossus of Rhodes collapse?",
    ]

    for question in example_questions:
        print(f"======={question}=============")
        print_answer(basic_rag_pipeline, question)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
