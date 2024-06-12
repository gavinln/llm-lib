"""
https://github.com/guidance-ai/guidance/blob/main/notebooks/api_examples/models/OpenAI.ipynb

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

from guidance import assistant, gen, instruction, models, system, user

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def main():
    print("-----instruct usage--------")
    gpt35_instruct = models.OpenAI(  # type: ignore
        "gpt-3.5-turbo-instruct", echo=False
    )

    lm = gpt35_instruct
    with instruction():
        lm += "What is a popular flavor?"
    lm += gen("flavor", max_tokens=10, stop=".")
    print(lm["flavor"])

    print("-----chat usage------------")
    gpt35 = models.OpenAI("gpt-3.5-turbo", echo=False)  # type: ignore

    lm = gpt35

    with system():
        lm += "You only speak in ALL CAPS."

    with user():
        lm += "What is the captial of Greenland?"

    with assistant():
        lm += gen("answer", max_tokens=20)

    print(lm["answer"])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
