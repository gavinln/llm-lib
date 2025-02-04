"""
Evaluate RAG with LlamaIndex
https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex
"""

import asyncio
import datetime as dt
import logging
import pathlib
import tempfile
from typing import Any, List

import llama_index
import pandas as pd
from joblib import Memory
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)


def get_data_dir():
    return str(pathlib.Path(SCRIPT_DIR / "data" / "paul_graham"))


def get_temp_dir():
    temp_dir = pathlib.Path(SCRIPT_DIR / "temp")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_qa_dataset_file() -> pathlib.Path:
    qa_dataset_file = get_temp_dir() / "qa_dataset_temp.txt"
    return qa_dataset_file


@memory.cache
def get_query_response(query: str) -> Any:

    documents: List[Document] = SimpleDirectoryReader(
        get_data_dir()
    ).load_data()
    # Build index with a chunk_size of 512
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"There are {len(nodes)} nodes in the documents")
    vector_index = VectorStoreIndex(nodes)
    query_engine: BaseQueryEngine = vector_index.as_query_engine()
    query = "What did the author do growing up?"
    response = query_engine.query(query)
    print(f"There are {len(response.source_nodes)} nodes in the response")
    return vector_index, nodes, response


"""
Hit Rate:

Hit rate calculates the fraction of queries where the correct answer is found
within the top-k retrieved documents. In simpler terms, it’s about how often
our system gets it right within the top few guesses.

Mean Reciprocal Rank (MRR):

For each query, MRR evaluates the system’s accuracy by looking at the rank of
the highest-placed relevant document. Specifically, it’s the average of the
reciprocals of these ranks across all the queries. So, if the first relevant
document is the top result, the reciprocal rank is 1; if it’s second, the
reciprocal rank is 1/2, and so on.
"""


def get_eval_results_df(eval_results):
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame({"Hit Rate": [hit_rate], "MRR": [mrr]})
    return metric_df


def clear_cache_memory():
    "run this to clear temporary files"
    memory.clear()
    qa_dataset_file = get_qa_dataset_file()
    if qa_dataset_file.exists():
        qa_dataset_file.unlink()


def main():
    llm = OpenAI(model="gpt-4o-mini")

    query = "What did the author do growing up?"
    vector_index, nodes, response = get_query_response(query)

    qa_dataset_file = get_qa_dataset_file()
    if qa_dataset_file.exists():
        print(f"Temporary dataset {qa_dataset_file} exists")
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(str(qa_dataset_file))
    else:
        print(f"Creating new temporary dataset {qa_dataset_file}")
        qa_dataset = generate_question_context_pairs(
            nodes, llm=llm, num_questions_per_chunk=2
        )
        qa_dataset.save_json(str(qa_dataset_file))

    retriever = vector_index.as_retriever(similarity_top_k=2)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = asyncio.run(
        retriever_evaluator.aevaluate_dataset(qa_dataset)
    )
    metric_df = get_eval_results_df(eval_results)
    print(metric_df)
    """
       Hit Rate       MRR
    0  0.762295  0.639344
    """


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    # clear_cache_memory()
    main()
