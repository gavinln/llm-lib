"""
Search the web using DuckDuckGo and OPENAI
https://docs.phidata.com/introduction
"""

import logging
import pathlib

from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    assistant = Assistant(tools=[DuckDuckGo()], show_tool_calls=True)
    query = "What's happening in France?"
    print(f"{query=}")
    assistant.print_response(query, markdown=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
