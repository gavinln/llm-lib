"""
DSPy building blocks
https://dspy-docs.vercel.app/docs/category/dspy-building-blocks
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


def direct_llm_call():
    "Call llm directly - not recommended"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)
    question = "What is the most popular attraction in the city of lights?"
    answer = gpt3_turbo(question)
    print_question_answer(question, answer)


def dspy_llm_call():
    "call llm with dspy signature"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)
    qa = dspy.ChainOfThought("question -> answer")
    question = "What is the most popular attraction in the city of lights?"
    response = qa(question=question)
    print_question_answer(question, response.answer)


def multiple_llms():
    "call multiple llms"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)
    question = (
        "What is the name of the capital city "
        + "to the north west of the city of lights?"
    )

    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question=question)
    print(gpt3_turbo.kwargs["model"])
    print_question_answer(question, response.answer)

    gpt4_turbo = dspy.OpenAI(model="gpt-4-turbo", max_tokens=300)
    with dspy.context(lm=gpt4_turbo):
        print(gpt4_turbo.kwargs["model"])
        response = qa(question=question)
        print_question_answer(question, response.answer)


def multiple_outputs():
    "generate multiple outputs"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)
    question = (
        "What is the name of the capital city "
        + "to the north west of the city of lights?"
    )

    qa = dspy.ChainOfThought("question -> answer", n=5)
    response = qa(question=question)
    print(gpt3_turbo.kwargs["model"])
    print_question_answer(question, response.completions.answer)


def main():
    fire.Fire(
        {
            "direct-llm-call": direct_llm_call,
            "dspy-llm-call": dspy_llm_call,
            "multiple-llms": multiple_llms,
            "multiple-outputs": multiple_outputs,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
