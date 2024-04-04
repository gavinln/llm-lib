import logging
import pathlib
from ast import literal_eval
from contextlib import contextmanager

import fire
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

tqdm.pandas(desc="progress...")


def get_food_reviews_file():
    return SCRIPT_DIR / "data" / "food_reviews_1k.csv"


def get_food_reviews_embedding_file():
    return SCRIPT_DIR / "output" / "food_reviews_embedding_1k.csv"


def get_food_reviews_dataset():
    df = pd.read_csv(get_food_reviews_file(), index_col=0)
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = (
        "Title: "
        + df.Summary.str.strip()
        + "; Content: "
        + df.Text.str.strip()
    )
    return df


def get_ag_news_file():
    return SCRIPT_DIR / "data" / "AG_news_samples.csv"


def get_ag_news_dataset():
    df = pd.read_csv(get_ag_news_file())
    return df


def get_food_reviews_embedding_dataset():
    df = pd.read_csv(get_food_reviews_embedding_file())
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    return df


@contextmanager
def temp_log_level(level):
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(old_level)


def get_text_embeddings(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return (
        OpenAI().embeddings.create(input=[text], model=model).data[0].embedding
    )


def save_food_reviews_embedding():
    "get and save the food reviews embeddings"
    df = get_food_reviews_dataset()
    with temp_log_level(logging.WARNING):
        df["embedding"] = df.combined.progress_apply(
            lambda x: get_text_embeddings(x)
        )
        output_file = get_food_reviews_embedding_file()
        output_dir = output_file.parent
        if not output_dir.exists():
            output_dir.mkdir()
        df.to_csv(output_file, index=False)


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def reduce_embeddings_dim():
    "reduce embeddings by truncating and normalizing"
    response = OpenAI().embeddings.create(
        model="text-embedding-3-small",
        input="Testing 123",
        encoding_format="float",
    )
    embedding = np.array(response.data[0].embedding)
    print(f"original dim {embedding.size}")
    cut_dim = embedding[:256]
    norm_dim = normalize_l2(cut_dim)
    print(f"normalized dim {norm_dim.size}")


def wikipedia_2022_olympics_curling():
    article = """

    Men Gold medal:

    Niklas Edin
    Oskar Eriksson
    Rasmus Wranå
    Christoffer Sundgren
    Daniel Magnusson

    Men Silver medal:

    Bruce Mouat
    Grant Hardie
    Bobby Lammie
    Hammy McMillan Jr.
    Ross Whyte

    Men Bronze medal:

    Brad Gushue
    Mark Nichols
    Brett Gallant
    Geoff Walker
    Marc Kennedy

    Women Gold medal:

    Eve Muirhead
    Vicky Wright
    Jennifer Dodds
    Hailey Duff
    Mili Smith

    Women Silver medal:

    Satsuki Fujisawa
    Chinami Yoshida
    Yumi Suzuki
    Yurika Yoshida
    Kotomi Ishizaki

    Women Bronze medal:

    Anna Hasselborg
    Sara McManus
    Agnes Knochenhauer
    Sofia Mabergs
    Johanna Heldin

    """
    return article


def answer_question_using_context():
    "provide context article to answer question"
    wikipedia_article_on_curling = wikipedia_2022_olympics_curling()

    query = f"""

    Use the below article on the 2022 Winter Olympics to answer the
    subsequent question. If the answer cannot be found, write "I don't know."

    Article:
    \"\"\"
    {wikipedia_article_on_curling}
    \"\"\"

    Question: Which male and female athletes won the gold medal in curling at
    the 2022 Winter Olympics?
    """

    completion = OpenAI().chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Answer questions about the 2022 Winter Olympics.",
            },
            {"role": "user", "content": query},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )

    print(completion.choices[0].message.content)


def search_reviews(df, product_description, n=3):
    product_embedding = get_text_embeddings(product_description)
    df["similarity"] = df.embedding.apply(
        lambda x: 1 - cosine(x, product_embedding)
    )

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    return results


def text_search_using_embeddings():
    "search reviews using embeddings"
    df = get_food_reviews_embedding_dataset()

    product_description = "delicious beans"
    print(f"searching for {product_description} {'-' * 10}")
    results = search_reviews(df, product_description)
    for r in results:
        print(r)
        print()

    product_description = "whole wheat pasta"
    print(f"searching for {product_description} {'-' * 10}")
    results = search_reviews(df, product_description)
    for r in results:
        print(r)
        print()


def recommend_using_embeddings():
    print("in function")
    df = get_ag_news_dataset()
    print(df.head())


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "save-food-reviews-embedding": save_food_reviews_embedding,
            "reduce-embeddings-dim": reduce_embeddings_dim,
            "answer-question-using-context": answer_question_using_context,
            "text-search-using-embeddings": text_search_using_embeddings,
            "recommend-using-embeddings": recommend_using_embeddings,
        }
    )


if __name__ == "__main__":
    main()
