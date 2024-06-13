"""
https://github.com/guidance-ai/guidance?tab=readme-ov-file#constrained-generation

https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/react.ipynb

OPENAI not supported for constrained generation as it does not have explicit
guidance integration
"""

import logging
import pathlib
import re
import sys

import fire
import guidance
from guidance import gen, models, select

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
def react_prompt_example(lm, question, max_rounds=10):
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
    print("in basics")
    query = "What is the temperature in the capital of France?"
    lm = get_model() + react_prompt_example(query)  # type: ignore
    print(lm)


def tools():
    "ReAct - reasoning & action using tools"
    print("in basics")
    query = "What is the temperature in the capital of France?"
    lm = get_model() + react_prompt_example(query)  # type: ignore
    print(lm)


def main():
    fire.Fire(
        {
            "stateful-basics": basics,
            "stateful-re-act": re_act,
            "stateful-tools": tools,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
