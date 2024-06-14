"""
https://haystack.deepset.ai/overview/quick-start

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

from pprint import pprint

from haystack import Pipeline, PredefinedPipeline

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def main():
    pipeline = Pipeline.from_template(PredefinedPipeline.CHAT_WITH_WEBSITE)
    if isinstance(pipeline, Pipeline):
        result = pipeline.run(
            {
                "fetcher": {
                    "urls": [
                        "https://haystack.deepset.ai/overview/quick-start"
                    ]
                },
                "prompt": {
                    "query": "Which components do I need for a RAG pipeline?"
                },
            }
        )
        print(result["llm"]["replies"])
        pprint(result["llm"]["meta"])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
