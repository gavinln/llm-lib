import logging
import os
import pathlib
from typing import Any

import numpy as np
import openai
import pandas as pd
from scipy.spatial.distance import cosine


from contextlib import contextmanager

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_embeddings(embedding_file):
    df = pd.read_csv(embedding_file, index_col=0)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
    return df


def create_context(question, df, max_len=3600):
    """
    Create a context for a question by finding the most similar
    context from the dataframe
    """

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Get the embeddings for the question
    q_embeddings = (
        client.embeddings.create(
            input=question, model="text-embedding-ada-002"
        )
        .data[0]
        .embedding
    )

    # Get the distances from the embeddings
    df["similarities"] = df.embeddings.apply(lambda x: cosine(x, q_embeddings))

    cur_len = 0
    context_doc = []

    # Sort by distance and add the text to the context
    # until the context is too long
    for _, row in df.sort_values("similarities", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        context_doc.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(context_doc)


def answer_question(
    df,
    question="Am I allowed to publish to Twitter, without a human review?",
    max_len=3600,
    debug=False,
    max_tokens=150,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    system_context = (
        "Answer the question based on the context below, and if the "
        "question can't be answered based on the context, say "
        '"I don\'t know"\n\n'
    )

    user_context = (
        f"Context: {context}\n\n---\n\n" f"Question: {question}\nAnswer:"
    )
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_context,
                },
                {
                    "role": "user",
                    "content": user_context,
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content.strip()
        # return response.choices[0].message.strip()
    except Exception as e:
        print(e)
        return ""


@contextmanager
def temp_log_level(level):
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(old_level)


def main():
    embedding_file = SCRIPT_DIR / "processed" / "embeddings.csv"
    df = get_embeddings(embedding_file)

    with temp_log_level(logging.WARNING):
        print("Example questions and answers")
        question = "What day is it?"
        answer = answer_question(df, question, debug=False)
        print(question + '\n' + answer)

        question = "What is our newest embeddings model?"
        answer = answer_question(df, question)
        print(question + '\n' + answer)

        question = "What is ChatGPT?"
        answer = answer_question(df, question)
        print(question + '\n' + answer)

        while True:
            user_input = input("Ask a question (type ENTER to quit): ")
            if user_input == "":
                break
            else:
                answer = answer_question(df, user_input)
                print(answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
