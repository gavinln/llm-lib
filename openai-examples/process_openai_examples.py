"""
print all tags from the openai examples list
"""

import logging
import pathlib
import sys

import fire
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_data_file():
    return SCRIPT_DIR / "openai-examples-list.csv"


def get_data():
    data_file = get_data_file()
    if not data_file.exists():
        sys.exit(f"Cannot find file {data_file}")
    df = pd.read_csv(data_file, sep="|", header=None)
    df.columns = ["title", "tags", "date"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_remaining_tag(tag_combo: str, tag: str):
    start_idx = tag_combo.find(tag)
    tag_rem = tag_combo
    if start_idx >= 0:
        tag_rem = tag_combo[0:start_idx] + tag_combo[start_idx + len(tag) :]
    return tag_rem


def get_remaining_tags(default_tags: list, tag_combos: list):
    "returns tags not in default_tags"
    tag_rems = []
    for tag_combo in tag_combos:
        tag_rem = tag_combo
        for tag in default_tags:
            tag_rem = get_remaining_tag(tag_rem, tag)
        if len(tag_rem) > 0:
            tag_rems.append(tag_rem)
    return list(set(tag_rems))


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


def print_tags():
    log.info("In process-file.py")
    df = get_data()
    unique_tag_combos: list = list(set(df.tags.tolist()))
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
    tags = get_remaining_tags(default_tags, unique_tag_combos)
    print("default tags------------\n", default_tags)
    print("new tags----------------\n", tags)

    tag_combos = []
    for tag_combo in df.tags.tolist():
        split_tags = get_split_tags(tag_combo, default_tags)
        tag_combos.append(split_tags)

    df["split_tags"] = tag_combos
    print(df)


def process_examples():
    df = get_data()
    print(df[df.title.str.contains('Unit')])


def main():
    fire.Fire({"print-tags": print_tags, "process-examples": process_examples})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
