"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib

from langchain_community.llms import Ollama

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    llm = Ollama(model="llama2")
    result: str = llm.invoke("Why is the sky blue?")
    print(result)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
