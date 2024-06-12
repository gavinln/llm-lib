"""
https://guidance.readthedocs.io/en/latest/example_notebooks/tutorials/intro_to_guidance.html

OPENAI_API_KEY needed
"""

import logging
import pathlib
import sys

import fire
import guidance
from guidance import gen, models, select

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_model():
    gpt3_5 = models.OpenAI(  # type: ignore
        "gpt-3.5-turbo-instruct", echo=False
    )
    return gpt3_5


def simple_gen():
    lm = get_model()
    query = "Who won the last Kentucky derby and by how much?"
    lm += query
    lm += gen(max_tokens=10)  # type: ignore
    print(lm)

    lm = get_model()
    lm += """
    Q: Who won the last Kentucky derby and by how much?
    A: """ + gen(
        stop="Q:"
    )  # type: ignore
    print(lm)


def templates():
    "use f-strings for templates"
    query = "Who won the last Kentucky derby and by how much?"

    lm = get_model()
    lm += f"""
    Q: {query}
    A: {gen(stop="Q:")}
    """
    print(lm)


def capturing():
    lm = get_model()
    query = "Who won the last Kentucky derby and by how much?"

    lm += f"""
    Q: {query}
    A: {gen(name="answer", stop="Q:")}
    """
    print(lm["answer"])


@guidance  # type: ignore
def qa_bot(lm, query):
    lm += f"""
    Q: {query}
    A: {gen(stop="Q:")}
    """
    return lm


def functions():
    lm = get_model()
    query = "Who won the last Kentucky derby and by how much?"
    lm += qa_bot(query)  # type: ignore
    print(lm)


def main():
    fire.Fire(
        {
            "basics-simple-gen": simple_gen,
            "basics-templates": templates,
            "basics-capturing": capturing,
            "basics-functions": functions,
        }
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
