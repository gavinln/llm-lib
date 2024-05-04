"""
https://cookbook.openai.com/examples/embedding_long_inputs
"""

import logging
import pathlib

from itertools import islice

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def test_batched():
    for item in batched('ABCDEFG', 3):
        print(item)


def main():
    test_batched()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
