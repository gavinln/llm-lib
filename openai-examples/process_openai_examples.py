import logging
import pathlib
import sys

import fire
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_file():
    return SCRIPT_DIR / "openai-examples-list.csv"


def get_openai_examples_data():
    "get openai examples from a csv file as a dataframe"
    data_file = get_data_file()
    if not data_file.exists():
        sys.exit(f"Cannot find file {data_file}")
    df = pd.read_csv(data_file, sep="|", header=None)
    df.columns = ["title", "tags", "date"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_remaining_tag(tag_combo: str, tag: str):
    "remove tags from a tag_combo and returns remaining"
    start_idx = tag_combo.find(tag)
    tag_rem = tag_combo
    if start_idx >= 0:
        tag_rem = tag_combo[0:start_idx] + tag_combo[start_idx + len(tag) :]
    return tag_rem


def test_get_remaining_tag():
    tag_combo = "completionstiktokenembeddings"
    tag_rem1 = get_remaining_tag(tag_combo, "completions")
    tag_rem2 = get_remaining_tag(tag_combo, "tiktoken")
    tag_rem3 = get_remaining_tag(tag_combo, "embeddings")
    assert (tag_rem1, tag_rem2, tag_rem3) == (
        "tiktokenembeddings",
        "completionsembeddings",
        "completionstiktoken",
    )


def get_new_tags(default_tags: list, tag_combos: list):
    "returns tags not in default_tags"
    tag_rems = []
    for tag_combo in tag_combos:
        tag_rem = tag_combo
        for tag in default_tags:
            tag_rem = get_remaining_tag(tag_rem, tag)
        if len(tag_rem) > 0:
            tag_rems.append(tag_rem)
    return list(set(tag_rems))


def test_get_new_tags():
    default_tags = ["completions", "tiktoken"]
    missing_tag = "embeddings"
    tag_combos = ["".join(default_tags) + missing_tag]
    tag_rems = get_new_tags(default_tags, tag_combos)
    assert tag_rems == [missing_tag]


def get_split_tags(tag_combo: str, tags: list):
    split_tags = []
    tag_rem = tag_combo
    for tag in tags:
        start_len = len(tag_rem)
        tag_rem = get_remaining_tag(tag_rem, tag)
        end_len = len(tag_rem)
        if start_len != end_len:
            split_tags.append(tag)
    return split_tags


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

    tag_combos = []
    for tag_combo in df.tags.tolist():
        split_tags = get_split_tags(tag_combo, default_tags)
        tag_combos.append(split_tags)

    df["split_tags"] = tag_combos
    print(df)


def process_examples():
    df = get_openai_examples_data()
    print(df[df.title.str.contains("Unit")])


def main():
    fire.Fire({"print-tags": print_tags, "process-examples": process_examples})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
