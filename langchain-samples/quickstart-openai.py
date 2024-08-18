"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib

from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    llm = ChatOpenAI()
    query = "How can langsmith help with testing? Be concise."
    bm1: BaseMessage = llm.invoke(query)
    print(bm1.content)
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class technical documentation writer.",
            ),
            ("user", "{input}"),
        ]
    )
    chain = prompt | llm
    bm2: BaseMessage = chain.invoke({"input": query})
    print(bm2.content)

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    content: str = chain.invoke({"input": query})
    print(content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
