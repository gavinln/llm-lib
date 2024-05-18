"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    llm = Ollama(model="orca-mini")

    # use default system prompt
    # ask a question that is in the training data
    # result: str = llm.invoke("Why is the sky blue?")
    # print(result)

    query: str = "how can langsmith help with testing?"
    result: str = llm.invoke(query)
    print(result)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class technical documentation writer."
                " Be concise",
            ),
            ("user", "{input}"),
        ]
    )
    chain = prompt | llm
    result: str = chain.invoke(query)  # type: ignore
    breakpoint()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
