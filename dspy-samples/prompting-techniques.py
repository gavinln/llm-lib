"""
DSPy prompting techniques
https://dspy-docs.vercel.app/docs/building-blocks/modules
"""

import logging
import pathlib

import fire
import dspy

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def print_question_answer(question, answer):
    print(f"Question: {question}")
    print(f"Answer: {answer}")


def inline_signatures():
    "inline signatures defined with a string"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)

    question = "What is the capital of France"
    qa = dspy.Predict("question -> answer")
    response = qa(question=question)
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")


def predict_prompt():
    "basic prompt"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)

    sentence = "it's a charming and affecting journey."
    classify = dspy.Predict("sentence -> sentiment")
    response = classify(sentence=sentence)
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")


def chain_of_thought_prompt():
    "chain-of-thought prompt"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)

    question = "What's great about the ColBERT retrieval model?"
    qa = dspy.ChainOfThought("question -> answer", n=3)
    response = qa(question=question)
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")

    for k, v in response.completions.items():
        print(f"lists {k}:----------------\n{v}")


def react_prompt():
    pass


def multi_chain_comparison_prompt():
    pass


def main():
    fire.Fire(
        {
            "predict-prompt": predict_prompt,
            "chain-of-thought-prompt": chain_of_thought_prompt,
            "react-prompt": react_prompt,
            "multi-chain-comparison-prompt": multi_chain_comparison_prompt,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
