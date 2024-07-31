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


def inline_signatures():
    "inline signatures defined with a string"
    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)

    question = "What is the capital of France"
    qa = dspy.Predict("question -> answer")
    response = qa(question=question)
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")

    sentence = "It's a charming and often affecting journey."
    classify = dspy.Predict("sentence -> sentiment")
    sentiment = classify(sentence=sentence)
    for k, v in sentiment.items():
        print(f"{k}:----------------\n{v}")

    document = """
    Paris is the capital of France. It is the largest city in France. It has an
    estimated population of 2,102,650 residents and an area of more than 105
    km2.
    """

    summarize = dspy.Predict("document -> summary")
    response = summarize(document=document.replace("\n", " "))
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")

    qa = dspy.ChainOfThought("question -> answer")
    question = "What is the top tourist attraction in the city of lights?"
    response = qa(question=question)
    for k, v in response.items():
        print(f"{k}:----------------\n{v}")


def class_signatures():
    "class signatures defined using a class"

    gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=gpt3_turbo)

    class Emotion(dspy.Signature):
        "Classify emotion among sadness, joy, love, anger, fear, surprise."

        sentence = dspy.InputField()
        sentiment = dspy.OutputField()

    sentence = (
        "i started feeling a little vulnerable "
        "when the giant spotlight started blinding me"
    )

    classify = dspy.Predict(Emotion)
    response = classify(sentence=sentence)

    for k, v in response.items():
        print(f"{k}:----------------\n{v}")

    class CheckCitationFaithfulness(dspy.Signature):
        "Verify that the text is based on the provided context."

        context = dspy.InputField(desc="facts here are assumed to be true")
        text = dspy.InputField()
        faithfulness = dspy.OutputField(
            desc="True/False indicating if text is faithful to context"
        )

    context = """
    The 21-year-old made seven appearances for the Hammers and netted his only
    goal for them in a Europa League qualification round match against Andorran
    side FC Lustrains last season. Lee had two loan spells in League One last
    term, with Blackpool and then Colchester United. He scored twice for the
    U's but was unable to save them from relegation. The length of Lee's
    contract with the promoted Tykes has not been revealed. Find all the latest
    football transfers on our dedicated page.
    """

    text = "Lee scored 3 goals for Colchester United."

    faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    response = faithfulness(context=context.replace("\n", " "), text=text)

    for k, v in response.items():
        print(f"{k}:----------------\n{v}")


def main():
    fire.Fire(
        {
            "inline-signatures": inline_signatures,
            "class-signatures": class_signatures,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
