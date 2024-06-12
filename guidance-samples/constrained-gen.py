"""
https://github.com/guidance-ai/guidance?tab=readme-ov-file#constrained-generation

OPENAI not supported for constrained generation as it does not have explicit
guidance integration
"""

import logging
import pathlib
import sys

import fire
import guidance
from guidance import gen, models, one_or_more, select, system, zero_or_more

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_model():
    # return models.Transformers("gpt2", echo=False)  # type: ignore
    return models.Transformers(  # type: ignore
        "pankajmathur/orca_mini_3b", echo=False
    )


def select_constrained_gen():
    gpt2 = get_model()
    lm = gpt2 + "I like the color "
    lm += select(["red", "blue", "green"])
    print(lm)


@guidance  # type: ignore
def qa_bot(lm, query):
    lm += f"""
    Q: {query}
    Now I will choose to either SEARCH the web or RESPOND.
    Choice: {select(["SEARCH", "RESPOND"], name="choice")}
    """
    if lm["choice"] == "SEARCH":
        lm += "A: I don't know. Google it!"
    else:
        lm += "A: {}".format(
            gen(stop="Q:", name="answer", max_tokens=30)  # type: ignore
        )
    return lm


def interleave_constrained_gen():
    query = "Who won the last Kentucky derby and by how much?"
    lm = get_model()
    lm += qa_bot(query)  # type: ignore
    print(lm)


def regex_constrained_gen():
    query = "Question: Luke has ten balls. He gives three to his brother.\n"
    lm = get_model() + query
    lm += "How many balls does he have left?\n"
    lm += "Answer: " + gen(regex=r"\d+")  # type: ignore
    print(lm)


@guidance(stateless=True)  # type: ignore
def number(lm):
    n = one_or_more(
        select(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    )  # type: ignore
    # Allow for negative or positive numbers
    return lm + select(["-" + n, n])


@guidance(stateless=True)  # type: ignore
def operator(lm):
    return lm + select(["+", "*", "**", "/", "-"])


@guidance(stateless=True)  # type: ignore
def expression(lm):
    """
    Either
    1. A number (terminal)
    2. two expressions with an operator and optional whitespace
    3. An expression with parentheses around it
    """
    return lm + select(
        [
            number(),
            expression()
            + zero_or_more(" ")
            + operator()
            + zero_or_more(" ")
            + expression(),
            "(" + expression() + ")",
        ]
    )


def cfg_constrained_gen():
    "context free grammar"
    query = "Problem: What is three plus two?" "\n"
    lm = get_model() + query
    lm += "Equivalent arithmetic expression: " + gen(
        max_tokens=30, stop="\n"
    )  # type: ignore
    print(lm)

    lm = get_model() + query
    lm += "Equivalent arithmetic expression: " + expression()  # type: ignore
    print(lm)


def main():
    fire.Fire(
        {
            "constrained-select": select_constrained_gen,
            "constrained-interleave": interleave_constrained_gen,
            "constrained-regex": regex_constrained_gen,
            "constrained-cfg": cfg_constrained_gen,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
