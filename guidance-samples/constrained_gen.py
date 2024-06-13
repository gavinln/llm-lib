"""
https://github.com/guidance-ai/guidance?tab=readme-ov-file#constrained-generation

OPENAI not supported for constrained generation as it does not have explicit
guidance integration
"""

import logging
import pathlib
import re
import sys

import fire
import guidance
from guidance import gen, models, one_or_more, select, system, zero_or_more

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))

_LOADED_MODEL = None


def get_model():
    global _LOADED_MODEL
    if not _LOADED_MODEL:
        _LOADED_MODEL = models.Transformers(  # type: ignore
            "pankajmathur/orca_mini_3b", echo=False
        )
    return _LOADED_MODEL


def select_constrained_gen():
    lm = get_model()
    lm += "I like the color "
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
    model = get_model()
    lm = model + query
    lm += "Equivalent arithmetic expression: " + gen(
        max_tokens=30, stop="\n"
    )  # type: ignore
    print(lm)

    lm = model + query
    lm += "Equivalent arithmetic expression: " + expression()  # type: ignore
    print(lm)


@guidance(stateless=True)  # type: ignore
def ner_instruction(lm, input):
    lm += f"""
    Please tag each word in the input with PER, ORG, LOC, or nothing
    ---
    Input: John worked at Apple.
    Output:
    John: PER
    worked:
    at:
    Apple: ORG
    .:
    ---
    Input: {input}
    Output:
    """
    return lm


def cfg_constrained_gen2():
    "context free grammar2"
    input = "Julia never went to Morocco in her life!!"
    lm = get_model()
    lm += ner_instruction(input) + gen(stop="---")  # type: ignore
    print(lm)


@guidance(stateless=True)  # type: ignore
def constrained_ner(lm, input):
    # Split into words
    words = [
        x
        for x in re.split("([^a-zA-Z0-9])", input)
        if x and not re.match(r"\s", x)
    ]
    ret = ""
    for x in words:
        ret += x + ": " + select(["PER", "ORG", "LOC", ""]) + "\n"
    return lm + ret


def cfg_constrained_gen3():
    "context free grammar2"
    input = "Julia never went to Morocco in her life!!"
    lm = get_model()
    lm += ner_instruction(input) + constrained_ner(input)  # type: ignore
    print(lm)


def main():
    fire.Fire(
        {
            "constrained-select": select_constrained_gen,
            "constrained-interleave": interleave_constrained_gen,
            "constrained-regex": regex_constrained_gen,
            "constrained-cfg": cfg_constrained_gen,
            "constrained-cfg2": cfg_constrained_gen2,
            "constrained-cfg3": cfg_constrained_gen3,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
