"""
https://docs.activeloop.ai/examples/rag/quickstart

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

import openai
from deeplake.core.vectorstore import VectorStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_source_text():
    text_file = pathlib.Path(
        SCRIPT_DIR / "data" / "paul_graham" / "paul_graham_essay.txt"
    )
    return text_file.read_text()


def get_temp_storage_dir():
    return pathlib.Path(SCRIPT_DIR / "temp_storage")


def get_chunked_text(text: str, chunk_size: int = 1000):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_embeddings(texts: list[str] | str, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [
        item.embedding
        for item in openai.embeddings.create(input=texts, model=model).data
    ]


def main():
    source_text = get_source_text()
    chunked_text = get_chunked_text(source_text)
    vector_store = VectorStore(path=get_temp_storage_dir())
    vector_store.add(
        text=chunked_text,
        embedding_function=get_embeddings,
        embedding_data=chunked_text,
        metadata=[{"source": source_text}] * len(chunked_text),
    )
    prompt = "What are the first programs he tried writing?"
    search_results = vector_store.search(
        embedding_data=prompt, embedding_function=get_embeddings
    )
    print(search_results.keys())  # type: ignore
    print("-------prompt----------")
    print(prompt)
    print("-------text------------")
    for text in search_results["text"]:
        print(text[:300])
        break


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
