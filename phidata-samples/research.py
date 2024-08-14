"""
Search the web using DuckDuckGo and use top 3 links to write article with
OpenAI

https://docs.phidata.com/basics
"""

import logging
import pathlib

from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    assistant = Assistant(  # type: ignore
        tools=[DuckDuckGo(), Newspaper4k()],
        show_tool_calls=True,
        description=(
            "You are a senior NYT researcher writing an article on a topic."
        ),
        instructions=[
            "For the provided topic, search for the top 3 links.",
            "Then read each URL and extract the article text, ",
            "if a URL isn't available, ignore and let it be.",
            "Analyse and prepare an NYT worthy article based on ",
            "the information.",
        ],
        add_datetime_to_instructions=True,
    )
    query = "Latest developments in AI"
    print(f"{query=}")
    assistant.print_response(query, markdown=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
