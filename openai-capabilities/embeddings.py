import logging
import pathlib
from ast import literal_eval
from contextlib import contextmanager
from typing import Any

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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


def get_ag_news_embedding_file():
    return SCRIPT_DIR / "output" / "AG_news_samples_embedding.csv"


def get_ag_news_dataset():
    df = pd.read_csv(get_ag_news_file())
    return df


def get_food_reviews_embedding_dataset():
    df = pd.read_csv(get_food_reviews_embedding_file())
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    return df


def get_ag_news_embedding_dataset():
    df = pd.read_csv(get_ag_news_embedding_file())
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


def save_food_reviews_embeddings():
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


def save_ag_news_embeddings():
    "get and save the ag news embeddings"
    df = get_ag_news_dataset()
    with temp_log_level(logging.WARNING):
        df["embedding"] = df.description.progress_apply(
            lambda x: get_text_embeddings(x)
        )
        output_file = get_ag_news_embedding_file()
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


def search_reviews_text(df, product_description, n=3):
    product_embedding = get_text_embeddings(product_description)
    df["similarity"] = df.embedding.apply(
        lambda x: cosine(x, product_embedding)
    )

    results = (
        df.sort_values("similarity", ascending=True)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    return results


def search_reviews():
    "search reviews using embeddings"
    df = get_food_reviews_embedding_dataset()

    product_description = "delicious beans"
    print(f"searching for {product_description} {'-' * 10}")
    results = search_reviews_text(df, product_description)
    for r in results:
        print(r)
        print()

    product_description = "whole wheat pasta"
    print(f"searching for {product_description} {'-' * 10}")
    results = search_reviews_text(df, product_description)
    for r in results:
        print(r)
        print()


def recommendations_indices_from_strings(
    strings: list[str],
    embeddings: list[np.ndarray],
    source_string_index: int,
    k_nearest_neighbors: int = 1,
) -> list[Any]:
    "return nearest neighbor indices from smallest to largest"
    print("Number of strings:", len(strings))
    print("Number of embeddings:", len(embeddings))
    print("source string index:", source_string_index)
    print("Number of nearest neighbors:", k_nearest_neighbors)

    query_embedding = embeddings[source_string_index]

    distances = [
        (idx, cosine(embedding, query_embedding))
        for idx, embedding in enumerate(embeddings)
    ]
    sorted_distances = sorted(distances, key=lambda p: p[1])
    return sorted_distances[: min(len(distances), k_nearest_neighbors)]


def recommendation_news():
    "recommend similar news articles"
    df = get_ag_news_embedding_dataset()
    source_string_idx = 0
    print("Searching for {}".format(df.description[source_string_idx]))

    recommendation_indices = recommendations_indices_from_strings(
        df.description.tolist(), df.embedding.tolist(), 0, 5
    )

    indices = [r[0] for r in recommendation_indices]
    print("\n".join(df.description[indices].tolist()))


def visualization_reviews():
    "visualize distribution of reviews"
    df = get_food_reviews_embedding_dataset()

    # only use rating 1, 2, 3
    df_subset = df[df.Score.isin([1, 2, 3])]

    embedding_matrix = np.array(df_subset.embedding.tolist())

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        random_state=42,
        init="random",
        learning_rate=200.0,  # type: ignore
    )
    vis_dims = tsne.fit_transform(embedding_matrix)
    x, y = zip(*vis_dims)

    # colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
    colors = ["red", "gold", "darkgreen"]
    color_indices = df_subset.Score.values - 1

    colormap = matplotlib.colors.ListedColormap(colors)  # type: ignore

    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    legend = ax.legend(handles, labels, loc="upper right", title="Sizes")
    ax.add_artist(legend)

    for score in [0, 1, 2]:
        avg_x = np.array(x)[df_subset.Score - 1 == score].mean()
        avg_y = np.array(y)[df_subset.Score - 1 == score].mean()
        color = colors[score]
        ax.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    fig.suptitle("Amazon ratings visualized in language using t-SNE")
    plt.show()


def regression_reviews():
    "regression of reviews vs score"
    df = get_food_reviews_embedding_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedding.values), df.Score, test_size=0.2, random_state=42
    )

    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    preds = rfr.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")

    dr = DummyRegressor(strategy="mean")
    dr.fit(X_train, y_train)
    preds = dr.predict(X_test)

    mse_d = mean_squared_error(y_test, preds)
    mae_d = mean_absolute_error(y_test, preds)
    print(f"dummy performance: mse={mse_d:.2f}, mae={mae_d:.2f}")


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "save-food-reviews-embeddings": save_food_reviews_embeddings,
            "reduce-embeddings-dim": reduce_embeddings_dim,
            "answer-question-using-context": answer_question_using_context,
            "search-reviews": search_reviews,
            "save-ag-news-embeddings": save_ag_news_embeddings,
            "recommendation_news": recommendation_news,
            "visualization-reviews": visualization_reviews,
            "regression-reviews": regression_reviews,
        }
    )


if __name__ == "__main__":
    main()
