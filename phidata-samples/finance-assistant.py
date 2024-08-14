"""
Use the Yahoo finance API and OpenAI to write about finance
https://docs.phidata.com/introduction
"""

import logging
import pathlib

from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    assistant = Assistant(
        llm=OpenAIChat(model="gpt-4o"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                company_info=True,
                company_news=True,
            )
        ],
        show_tool_calls=True,
        markdown=True,
    )
    query = "What is the stock price of NVDA?"
    print(f"{query=}")
    assistant.print_response(query)
    query = "Write a comparison between NVDA and AMD, use all tools available."
    print(f"{query=}")
    assistant.print_response(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
