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
import pandas as pd

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


def clustering():
    np.random.seed(1234)  # make reproducible
    x = np.random.random((10_000, 100)).astype("float32")
    ncentroids = 1024
    niter = 20  # number of iterations
    verbose = True
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)
    print("Iteration stats")
    print(pd.DataFrame(kmeans.iteration_stats))
    print(f"{kmeans.centroids.shape=}")  # type: ignore

    # nearest centroid for each vector in x
    D, I = kmeans.index.search(x[:5], 1)  # type: ignore

    print(D.T)  # squared L2 distances
    print(I.T)

    chosen_centroids = kmeans.centroids[I.T].squeeze()
    l2_norm = np.linalg.norm(chosen_centroids - x[:5], axis=1)
    squared_L2_distances = l2_norm**2
    print(squared_L2_distances)

    print("squared distances should match")

    index = faiss.IndexFlatL2(d)
    index.add(x)  # type: ignore
    # 10 nearest points to centroids
    D, I = index.search(chosen_centroids, 10)
    print(I)


def pca():
    pass


def quantization():
    pass


def main():
    fire.Fire(
        {
            "clustering": clustering,
            "pca": pca,
            "quantization": quantization,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.INFO)
    main()
