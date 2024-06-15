"""
https://haystack.deepset.ai/overview/quick-start

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys
from pprint import pprint
from typing import Any

import fire
from haystack import Pipeline, PredefinedPipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_pipeline_config() -> dict[str, Any]:
    return {
        "fetcher": {
            "urls": ["https://haystack.deepset.ai/overview/quick-start"]
        },
        "prompt": {"query": "Which components do I need for a RAG pipeline?"},
    }


def website_chat():
    pipeline = Pipeline.from_template(PredefinedPipeline.CHAT_WITH_WEBSITE)
    if isinstance(pipeline, Pipeline):
        result = pipeline.run(get_pipeline_config())
        print(result["llm"]["replies"])
        pprint(result["llm"]["meta"])


def fetch_query():
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    prompt_template = """
    According to the contents of this website:
    {% for document in documents %}
        {{document.content}}
    {% endfor %}
    Answer the given question: {{query}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    pipeline = Pipeline()
    pipeline.add_component("fetcher", fetcher)
    pipeline.add_component("converter", converter)
    pipeline.add_component("prompt", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("fetcher.streams", "converter.sources")
    pipeline.connect("converter.documents", "prompt.documents")
    pipeline.connect("prompt.prompt", "llm.prompt")

    result = pipeline.run(get_pipeline_config())
    print(result["llm"]["replies"])
    pprint(result["llm"]["meta"])


def main():
    fire.Fire(
        {
            "starter-website-chat": website_chat,
            "starter-fetch-query": fetch_query,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
