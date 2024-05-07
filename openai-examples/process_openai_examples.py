"""

This module provides functions to process and display information about OpenAI
examples.

"""

import logging
import pathlib
import re
import sys
from collections import Counter

import fire
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_file():
    data_file = SCRIPT_DIR / "openai-examples-list.csv"
    if not data_file.exists():
        sys.exit(f"Cannot find file {data_file}")
    return data_file


def get_openai_examples_data():
    "get openai examples from a csv file as a dataframe"
    data_file = get_data_file()
    df = pd.read_csv(data_file, sep="|", header=None)
    df.columns = ["title", "tags", "date"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_new_tags(default_tags: list, tag_combos: list):
    "returns tags not in default_tags"
    tag_rems = []
    for tag_combo in tag_combos:
        for tag in default_tags:
            tag_combo = tag_combo.replace(tag, "")
        if len(tag_combo) > 0:
            tag_rems.append(tag_combo)
    return list(set(tag_rems))


def test_get_new_tags():
    default_tags = ["completions", "tiktoken"]
    missing_tag = "embeddings"
    tag_combos = ["".join(default_tags) + missing_tag]
    tag_rems = get_new_tags(default_tags, tag_combos)
    assert tag_rems == [missing_tag]


def get_split_tags(tag_combo: str, tags: list):
    return [tag for tag in tags if tag in tag_combo]


def test_get_split_tag():
    split_tags = get_split_tags("completionschat", ["completions", "chat"])
    assert split_tags == ["completions", "chat"]


def get_default_tags():
    default_tags = [
        "completions",
        "chat",
        "vision",
        "embeddings",
        "batch",
        "assistants",
        "vision",
        "dall-e",
        "speech",
        "tiktoken",
        "guardrails",
        "functions",
        "whisper",
        "moderation",
        "vision",
    ]
    return default_tags


def print_tags():
    "print default and new tags"
    df = get_openai_examples_data()
    unique_tag_combos: list = list(set(df.tags.tolist()))

    default_tags = get_default_tags()
    new_tags = get_new_tags(default_tags, unique_tag_combos)

    print("default tags------------\n", default_tags)
    print("new tags----------------\n", new_tags)


def get_title_words(title: pd.Series):
    "get word counts from titles"
    # remove punctutation from titles
    punctuation_re = re.compile("['\-/()<>:;,.?]+")
    clean_titles = title.str.lower().apply(lambda s: punctuation_re.sub("", s))
    words = " ".join(clean_titles.tolist()).split()
    word_counts = Counter(words)
    return word_counts


def get_stop_words():
    words = """

    with using for and search embeddings to a vector how database as the api
    question answering models chat function calling semantic vision completions
    & retrieval in module data generation getting started use weights biases
    augmented db prompt functions visualizing generative batch processing long
    create assistants sdk new your evaluate an build agent preview philosophy
    analysis places post-processing example document unit test writing
    multi-step hybrid context embedding filtered vectorize summarizing
    documents synthetic part clip improve multimodal evals moderation parse pdf
    docs tag caption images combine clothing app implement creating slides
    graph overview python fine tuning narrating video visual capabilities make
    outputs consistent seed parameter named entity recognition enrich text
    openapi specification nodejs monitoring deep lake similarity postgres
    automate aws tasks completion own ai preparation model fine-tune
    summarization task nearby leveraging google customer profiles addressing
    transcription misspellings vs enhancing transcriptions pre- techniques
    knowledge base call legacy dynamic masks segment anything running queries
    store vectors json tool-using wikipedia articles robust augmentation format
    inputs chatgpt content extraction q&a work large language cases related
    resources from around web texts that are longer than models maximum length
    count tokens

    """
    space_re = re.compile(r"\s\+")
    return space_re.sub(" ", words).split()


def print_examples_stats():
    "print statistics of the openai examples"
    df = get_openai_examples_data()
    default_tags = get_default_tags()
    tag_combos = df.tags.apply(
        lambda tag_combo: get_split_tags(tag_combo, default_tags)
    )
    df["split_tags"] = tag_combos
    del df["tags"]

    print(f"Total examples {df.shape[0]}")
    print("Examples by category")
    tag_counts = {
        tag: df.split_tags.apply(lambda x: tag in x).sum()
        for tag in default_tags
    }
    print(tag_counts)
    word_counts_all = get_title_words(df.title)
    stop_words = get_stop_words()
    word_counts = [word for word in word_counts_all if word not in stop_words]
    word_counts = [word for word in word_counts if not word.isnumeric()]

    print(word_counts)

    vector_titles = df.title[
        df.title.str.lower().str.contains("vector")
    ].tolist()
    print("\n".join(vector_titles))


def vector_in_titles():
    "print examples with vector in title"
    df = get_openai_examples_data()
    vector_titles = df.title[
        df.title.str.lower().str.contains("vector")
    ].tolist()
    print("\n".join(vector_titles))


def redis_in_titles():
    "print examples with redis in title"
    df = get_openai_examples_data()
    redis_titles = df.title[
        df.title.str.lower().str.contains("redis")
    ].tolist()
    print("\n".join(redis_titles))


def process_examples():
    df = get_openai_examples_data()
    print(df[df.title.str.contains("Unit")])


def main():
    fire.Fire(
        {
            "print-tags": print_tags,
            "process-examples": process_examples,
            "print-examples-stats": print_examples_stats,
            "vector-in-titles": vector_in_titles,
            "redis-in-titles": redis_in_titles,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
