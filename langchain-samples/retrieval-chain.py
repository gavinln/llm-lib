"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/
"""

import logging
import pathlib
from typing import Any

from bs4 import BeautifulSoup
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def example_html_doc():
    html_doc = """
    <html><head><title>The Dormouse's story</title></head>
    <body>
    <p class="title"><b>The Dormouse's story</b></p>

    <p class="story">Once upon a time there were three little
    sisters; and their names were
    <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
    <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
    <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>

    <p class="story">...</p>
    """
    return html_doc


def get_soup_example(html_doc: str):
    soup = BeautifulSoup(html_doc, "html.parser")
    return soup


def beautifulsoup_example():
    html_doc = example_html_doc()
    soup: Any = get_soup_example(html_doc)
    print(f"{soup.title=}")
    print(f"{soup.title.name=}")
    print(f"{soup.title.string=}")
    print(f"{soup.title.parent.name=}")
    print(f"{soup.find_all('a')=}")

    print("----get all links----------------")
    for link in soup.find_all("a"):
        print(link.get("href"))

    print("----get text----------------")
    print(soup.get_text())

    print("----prettify----------------")
    print(soup.prettify())


def main():
    # beautifulsoup_example()

    langsmith_url = "https://docs.smith.langchain.com/how_to_guides"
    loader = WebBaseLoader(langsmith_url)
    breakpoint()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
