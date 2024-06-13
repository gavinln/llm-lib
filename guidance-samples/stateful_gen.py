"""
https://github.com/guidance-ai/guidance?tab=readme-ov-file#constrained-generation

https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/react.ipynb

OPENAI not supported for constrained generation as it does not have explicit
guidance integration
"""

import logging
import math
import pathlib
import re
import sys

import fire
import guidance
from guidance import gen, models, regex, select

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


@guidance(stateless=False)  # type: ignore
def basics_(lm, query):
    lm += query + select(["yew", "no"], name="answer")
    if lm["answer"] == "yes":
        lm += "\nScott"
    else:
        lm += "\nNot Scott"
    return lm


def basics():
    print("in basics")
    query = "Should I say Scott?\n"
    lm = get_model() + basics_(query)  # type: ignore
    print(lm)


@guidance  # type: ignore
def react_prompt_example(lm, question, max_rounds=5):
    lm += f"Question: {question}\n"
    i = 1
    while True:
        lm += f"Thought {i}: " + gen(suffix="\n")  # type: ignore
        lm += f"Act {i}: " + select(["Search", "Finish"], name="act")
        lm += "[" + gen(name="arg", suffix="]") + "\n"  # type: ignore
        if lm["act"] == "Finish" or i == max_rounds:
            break
        else:
            lm += f"Observation {i}: "
            lm += gen(max_tokens=10) + "\n"  # type: ignore
        i += 1
    return lm


def re_act():
    "ReAct - reasoning & action"
    query = "What is the temperature in the capital of France?"
    lm = get_model() + react_prompt_example(query)  # type: ignore
    print(lm)


def get_tools_prompt():
    return """
    Answer the following questions as best you can. You have access only to the
    following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought 1: you should always think about what to do
    Action 1: the action to take, has to be one of {tool_names}
    Observation 1: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought N: I now know the final answer.
    Final Answer: the final answer to the original input question.
    Done.

    Example:
    Question: What is the square root of the age of Brad Pitt?
    Thought 1: I should find out how old Brad Pitt is.
    Action 1: age(Brad Pitt)
    Observation 1: 56
    Thought 2: I should find the square root of 56.
    Action 2: sqrt(56)
    Observation 2: 7.48
    Thought 3: I now know the final answer.
    Final Answer: 7.48
    Done.

    Question: {query}
    """


ages_db = {"Leonardo DiCaprio": 49, "Brad Pitt": 59}


@guidance  # type: ignore
def sqrt(lm, number):
    lm += (
        f'\nObservation {regex(r"[0-9]+")}: ' + f"{math.sqrt(float(number))}\n"
    )
    return lm


@guidance  # type: ignore
def log(lm, number):
    lm += f'\nObservation {regex(r"[0-9]+")}: {math.log(float(number)):.4f}\n'
    return lm


@guidance  # type: ignore
def age(lm, person):
    lm += f'\nObservation {regex(r"[0-9]+")}: {ages_db.get(person)}\n'
    return lm


tools = {
    "sqrt": "Computes the square root of a number.",
    "age": "Returns the age of a person.",
    "log": "Computes the logarithm of a number.",
}

tool_map = {
    "sqrt": lambda x: str(math.sqrt(float(x))),
    "age": lambda x: str(ages_db.get(x)),
    "log": lambda x: str(math.log(float(x))),
}


def tools1():
    "ReAct - reasoning & action using tools"
    query = "What is the logarithm of Leonardo DiCaprio's age?"
    prompt = get_tools_prompt()
    prompt_with_query = prompt.format(
        tools=tools, tool_names=list(tools.keys()), query=query
    )
    # DOES NOT work -----------------------------------------------------
    # assert token_byte_positions[-1] == last_pos, "Cross check last_pos"
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # AssertionError: Cross check last_pos
    # lm = get_model()
    # lm += prompt_with_query + gen(  # type: ignore
    #     max_tokens=200, tools=[sqrt, age, log], stop="Done."
    # )
    # print(lm)


@guidance  # type: ignore
def tools2_prompt_example(lm, question, tools, max_rounds=10):
    prompt = get_tools_prompt()
    tool_names = list(tools.keys())
    prompt_with_query = prompt.format(
        tools=tools, tool_names=tool_names, query=question
    )
    lm += prompt_with_query
    i = 1
    while True:
        lm += f"Thought {i}: " + gen(name="thought", suffix="\n")
        if "final answer" in lm["thought"] or i == max_rounds:
            lm += "Final Answer: " + gen(name="answer", suffix="\n")
            break
        lm += f"Act {i}: " + select(tool_names, name="act")
        lm += "(" + gen(name="arg", suffix=")") + "\n"
        if lm["act"] in tool_map:
            lm += f"Observation {i}: " + tool_map[lm["act"]](lm["arg"]) + "\n"
        i += 1
    return lm


def tools2():
    "ReAct - reasoning & action using tools"
    query = "What is the logarithm of Leonardo DiCaprio's age?"

    # DOES NOT work -----------------------------------------------------
    # assert token_byte_positions[-1] == last_pos, "Cross check last_pos"
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # AssertionError: Cross check last_pos
    # lm = get_model()
    # lm += tools2_prompt_example(query, tools)  # type: ignore
    # print(lm)


def main():
    fire.Fire(
        {
            "stateful-basics": basics,
            "stateful-re-act": re_act,
            "stateful-tools1": tools1,
            "stateful-tools2": tools2,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
