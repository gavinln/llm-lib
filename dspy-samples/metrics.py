"""
DSPy building blocks
https://dspy-docs.vercel.app/docs/building-blocks/metrics
"""

import logging
import pathlib
import tempfile

from joblib import Memory

import fire
import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)


def print_question_answer(question, answer):
    print(f"Question: {question}")
    print(f"Answer: {answer}")


@memory.cache
def get_hotpot_qa_dataset():
    train_size = 10
    dataset = HotPotQA(train_size=train_size, dev_size=0, test_size=0)
    return dataset


def validate_answer(example, pred, trace=None):
    answers = pred.answer.lower().split("\n")
    pred_answer = answers[-1].replace("answer:", "").strip()

    answer_contained = example.answer.lower() in pred_answer.lower()
    if trace is None:
        return float(answer_contained)
    else:
        return answer_contained


def metrics_basic():
    "call llm with dspy signature"
    gpt4_turbo = dspy.OpenAI(model="gpt-4-turbo", max_tokens=300)
    dspy.configure(lm=gpt4_turbo)

    dataset = get_hotpot_qa_dataset()
    # set inputs
    train_data = [row.with_inputs("question") for row in dataset.train]

    print(len(dataset.train), len(dataset.dev), len(dataset.test))

    qa = dspy.Predict("question -> answer")
    scores = []
    for row in train_data:
        pred = qa(**row.inputs())
        # print_question_answer(row["question"], row["answer"])
        score = validate_answer(row, pred)
        scores.append(score)

    print(scores)

    evaluator = Evaluate(
        devset=train_data,
        num_threads=1,
        display_progress=False,
        display_table=False,
        return_outputs=False,
    )
    result = evaluator(qa, metric=validate_answer, return_all_scores=True)
    print(result)


def main():
    fire.Fire({"metrics-basic": metrics_basic})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
