"""
https://haystack.deepset.ai/tutorials/29_serializing_pipelines
"""

import logging
import pathlib
import sys

import yaml
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from wikipedia.wikipedia import WikipediaPage

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_summary_template():
    template = """
        Please create a summary about the following topic:
        {{ topic }}
    """
    return template


def get_translate_template():
    template = """
        Please translate the following to French:
        {{ topic }}
    """
    return template


def main():
    template = get_summary_template()
    builder = PromptBuilder(template=template)
    llm = HuggingFaceLocalGenerator(
        model="google/flan-t5-large",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 150},
    )

    pipeline = Pipeline()
    pipeline.add_component(name="builder", instance=builder)
    pipeline.add_component(name="llm", instance=llm)

    pipeline.connect("builder", "llm")

    topic = "Climate change"
    result = pipeline.run(data={"builder": {"topic": topic}})
    print(result["llm"]["replies"][0])

    yaml_pipeline = pipeline.dumps()
    yaml_obj = yaml.safe_load(yaml_pipeline)

    yaml_obj["components"]["builder"]["init_parameters"][
        "template"
    ] = get_translate_template()

    yaml_new_pipeline = yaml.safe_dump(yaml_obj)
    new_pipeline = Pipeline.loads(yaml_new_pipeline)
    new_result = new_pipeline.run(data={"builder": {"topic": topic}})
    print(new_result["llm"]["replies"][0])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
