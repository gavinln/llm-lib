"""
https://cookbook.openai.com/examples/embedding_long_inputs
"""

import logging
import pathlib
from itertools import islice

import numpy as np
import openai
import tiktoken
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

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


def test_get_embedding():
    emb1 = get_embedding("AGI")
    enc_text = encode_text("AGI")
    emb2 = get_embedding(enc_text)
    assert np.allclose(emb1, emb2, atol=1e-3)

    emb3 = get_embedding(["A", "G", "I"])
    assert len(emb1) == len(emb3)
    assert emb1 != emb3


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
    assert items == (("A", "B"), ("C", "D"), ("E",))


def encode_text(text, encoding_name=EMBEDDING_ENCODING):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


def chunked_tokens(text, encoding_name, chunk_length):
    tokens = encode_text(text, encoding_name)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def test_chunked_tokens():
    text = "AGI " * 20
    chunk_lengths = [
        len(chunk) for chunk in chunked_tokens(text, EMBEDDING_ENCODING, 10)
    ]
    assert chunk_lengths == [10, 10, 10, 10, 1]


def get_embedding_chunks(
    text, model, max_tokens, enconding_name, average=False
):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, enconding_name, max_tokens):
        chunk_embedding = get_embedding(chunk, model)
        chunk_embeddings.append(chunk_embedding)
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(
            chunk_embeddings, axis=0, weights=chunk_lens
        )
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


def main():
    # test_batched()
    long_text = "AGI " * 5000
    try:
        _ = get_embedding(long_text)
    except openai.BadRequestError as err:
        print(err)

    chunk_embeddings = get_embedding_chunks(
        long_text, EMBEDDING_MODEL, EMBEDDING_CTX_LENGTH, EMBEDDING_ENCODING
    )
    print("chunk_embeddings: shape", np.array(chunk_embeddings).shape)

    chunk_embeddings_avg = get_embedding_chunks(
        long_text,
        EMBEDDING_MODEL,
        EMBEDDING_CTX_LENGTH,
        EMBEDDING_ENCODING,
        True,
    )
    print("chunk_embeddings_avg: shape", np.array(chunk_embeddings_avg).shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
