"""
https://github.com/facebookresearch/faiss/wiki/Getting-started
"""

import logging
import pathlib
import time
from typing import Any, NamedTuple

import faiss
import fire
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


class DataSizes(NamedTuple):
    d: int  # dimension
    nb: int  # database size
    nq: int  # number of queries


def init_data_sizes():
    np.random.seed(1234)  # make reproducible
    d = 64  # dimension
    nb = 100_000  # database size
    nq = 10_000  # number of queries
    return DataSizes(d, nb, nq)


def init_data(ds: DataSizes) -> Any:
    # add a translation along the first dimension that depends on the index
    xb = np.random.random((ds.nb, ds.d)).astype("float32")
    xb[:, 0] += np.arange(ds.nb) / 1000.0
    xq = np.random.random((ds.nq, ds.d)).astype("float32")
    xq[:, 0] += np.arange(ds.nq) / 1000.0
    return xb, xq


def brute_force_search():
    """
    The two matrices handle vectors of size d

    xb matrix (nb, d) for the database
        contains all the vectors that must be indexed
    xq matrix (nq, d) for the query vectors
        for nearest neighbors search
    """

    ds = init_data_sizes()
    xb, xq = init_data(ds)

    # build the index
    index = faiss.IndexFlatL2(ds.d)
    print(f"{index.is_trained=}")
    index.add(xb)  # type: ignore
    print(f"{index.ntotal=}")

    # search 4 nearest neighbors
    k = 4
    D, I = index.search(xb[:5], k)  # type: ignore
    print(I)
    print(D)

    start = time.time()
    D, I = index.search(xq, k)  # type: ignore     # actual search
    elapsed = time.time() - start
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    print(f"{elapsed=}")


def voronoi_search():
    ds = init_data_sizes()
    xb, xq = init_data(ds)

    # build the index
    quantizer = faiss.IndexFlatL2(ds.d)

    nlist = 100  # number of cells

    # inverted file with stored vectors
    index = faiss.IndexIVFFlat(quantizer, ds.d, nlist)
    assert not index.is_trained
    index.train(xb)  # type: ignore
    assert index.is_trained

    k = 4  # 4 nearest neighbors
    index.add(xb)  # add may be a bit slower as well
    D, I = index.search(xq, k)  # actual search, default cells to visit: 1
    print(I[-5:])  # neighbors of the 5 last queries

    start = time.time()
    index.nprobe = 10  # cells to visit. default is 1
    D, I = index.search(xq, k)  # actual search, default cells to visit: 1
    elapsed = time.time() - start
    print(I[-5:])  # neighbors of the 5 last queries
    print(D[-5:])  # neighbors of the 5 last queries
    print(f"{elapsed=}")


def low_memory_search():
    ds = init_data_sizes()
    xb, xq = init_data(ds)

    nlist = 100  # number of cells
    m = 8  # number of subquantizers
    quantizer = faiss.IndexFlatL2(ds.d)
    sub_vector_bits = 8  # number of bits encoding for each sub-vector
    index = faiss.IndexIVFPQ(quantizer, ds.d, nlist, m, sub_vector_bits)

    index.train(xb)  # type: ignore
    index.add(xb)  # type: ignore

    k = 4  # nearest neighbors
    D, I = index.search(xb[:5], k)  # type: ignore
    print(I)
    print(D)

    index.nprobe = 10  # cells to visit. default is 1
    start = time.time()
    D, I = index.search(xq, k)  # type: ignore
    elapsed = time.time() - start
    print(I[-5:])
    print(f"{elapsed=}")


def main():
    fire.Fire(
        {
            "brute-force-search": brute_force_search,
            "voronoi-search": voronoi_search,
            "low-memory-search": low_memory_search,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
