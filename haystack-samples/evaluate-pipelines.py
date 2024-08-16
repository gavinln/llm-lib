"""
https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines
"""

import logging
import pathlib
import sys
import tempfile
from dataclasses import dataclass

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation.eval_run_result import EvaluationRunResult
from joblib import Memory

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))

memory = Memory(tempfile.gettempdir(), verbose=0)


@dataclass
class EvalData:
    documents: list[Document]
    questions: list[str]
    ground_truth_answers: list[str]


@memory.cache
def get_data() -> EvalData:
    dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
    dataset = dataset.select(range(25))  # type: ignore
    all_documents = [
        Document(content=doc["context"]) for doc in dataset  # type: ignore
    ]  # type: ignore
    all_questions = [doc["instruction"] for doc in dataset]  # type: ignore
    all_ground_truth_answers = [
        doc["response"] for doc in dataset  # type: ignore
    ]
    return EvalData(all_documents, all_questions, all_ground_truth_answers)


@memory.cache
def get_document_embedder():
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return document_embedder


@memory.cache
def load_document_store(documents: list[Document]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()
    document_embedder = get_document_embedder()
    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.SKIP
    )
    indexing = Pipeline()
    indexing.add_component(
        instance=document_embedder, name="document_embedder"
    )
    indexing.add_component(instance=document_writer, name="document_writer")

    indexing.connect(
        "document_embedder.documents", "document_writer.documents"
    )

    indexing.run({"document_embedder": {"documents": documents}})
    return document_store


def get_rag_pipeline(document_store):
    template = """
        You have to answer the following question based on the
        given context information only.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
    """

    query_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("query_embedder", query_embedder)
    rag_pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store, top_k=3)
    )
    rag_pipeline.add_component(
        "prompt_builder", PromptBuilder(template=template)
    )
    rag_pipeline.add_component(
        "generator", OpenAIGenerator(model="gpt-3.5-turbo")
    )
    rag_pipeline.add_component("answer_builder", AnswerBuilder())

    rag_pipeline.connect("query_embedder", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "generator")
    rag_pipeline.connect("generator.replies", "answer_builder.replies")
    rag_pipeline.connect("generator.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    return rag_pipeline


def ask_question(rag_pipeline, question) -> tuple[str, list[Document]]:
    response = rag_pipeline.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )
    return (
        response["answer_builder"]["answers"][0].data,
        response["answer_builder"]["answers"][0].documents,
    )


def get_eval_pipeline():
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component(
        "sas_evaluator",
        SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    return eval_pipeline


def get_evaluation_result(eval_data, rag_answers, results):
    inputs = {
        "question": eval_data.questions,
        # "contexts": list([d.content] for d in ground_truth_docs),
        "contexts": list([d.content] for d in eval_data.documents),
        # "answer": list(ground_truth_answers),
        "answer": eval_data.ground_truth_answers,
        "predicted_answer": rag_answers,
    }

    evaluation_result = EvaluationRunResult(
        run_name="pubmed_rag_pipeline", inputs=inputs, results=results
    )
    evaluation_result.score_report()
    return evaluation_result


def main():
    eval_data = get_data()
    document_store = load_document_store(eval_data.documents)
    rag_pipeline = get_rag_pipeline(document_store)

    question = (
        "Do high levels of procalcitonin in the early phase "
        + "after pediatric liver transplantation indicate poor "
        + "postoperative outcome?"
    )
    answer, documents = ask_question(rag_pipeline, question)

    rag_answers = []
    retrieved_docs = []

    for question in eval_data.questions:
        answer, documents = ask_question(rag_pipeline, question)
        rag_answers.append(answer)
        retrieved_docs.append(documents)

    eval_pipeline = get_eval_pipeline()
    results = eval_pipeline.run(
        {
            "doc_mrr_evaluator": {
                "ground_truth_documents": list(
                    [d] for d in eval_data.documents
                ),
                "retrieved_documents": retrieved_docs,
            },
            "faithfulness": {
                "questions": eval_data.questions,
                "contexts": list([d.content] for d in eval_data.documents),
                "predicted_answers": rag_answers,
            },
            "sas_evaluator": {
                "predicted_answers": rag_answers,
                "ground_truth_answers": eval_data.ground_truth_answers,
            },
        }
    )
    evaluation_result = get_evaluation_result(eval_data, rag_answers, results)
    results_df = evaluation_result.to_pandas()
    print(results_df)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
