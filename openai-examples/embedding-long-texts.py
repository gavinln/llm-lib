"""
https://cookbook.openai.com/examples/embedding_long_inputs
"""

import logging
import pathlib
from itertools import islice

import openai
import tiktoken
from tenacity import (retry, retry_if_not_exception_type, stop_after_attempt,
                      wait_random_exponential)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return (
        openai.OpenAI()
        .embeddings.create(input=text_or_tokens, model=model)
        .data[0]
        .embedding
    )


def test_embedding():
    emb1 = get_embedding("AGI")
    emb2 = get_embedding(["A", "G", "I"])
    assert len(emb1) == len(emb2)
    assert emb1 != emb2


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def test_batched():
    items = tuple(batched("ABCDE", 2))
    assert items == (('A', 'B'), ('C', 'D'), ('E',))


def main():
    # test_batched()
    long_text = 'AGI ' * 5000
    try:
        emb = get_embedding(long_text)
    except openai.BadRequestError as err:
        print(err)
    breakpoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
