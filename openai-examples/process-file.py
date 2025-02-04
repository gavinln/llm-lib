import logging
import pathlib
from pathlib import Path

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def get_default_tags() -> list[str]:
    default_tags: list[str] = [
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


def find_tag_indices(joint_tags: str) -> list[int]:
    tag_idx_list: list[int] = []
    for tag in get_default_tags():
        tag_idx: int = joint_tags.find(tag)
        if tag_idx >= 0:
            tag_idx_list.append(tag_idx)
    return sorted(tag_idx_list)


def convert_tag_upper(joint_tag: str) -> str:
    tag_indices: list[int] = find_tag_indices(joint_tag)
    new_joint_tag: list[str] = []
    for idx, letter in enumerate(joint_tag):
        if idx in tag_indices:
            new_joint_tag.append(letter.upper())
        else:
            new_joint_tag.append(letter)
    return "".join(new_joint_tag)


def main():
    # Specify the path to the text file
    file_path = SCRIPT_DIR / "openai-examples-list.csv"

    # Check if the file exists
    if file_path.exists():
        # Open the file in read mode
        with file_path.open(mode="r") as file:
            # Read the file line by line
            for line in file:

                pipe_first = line.find("|")
                pipe_last = line.rfind("|")
                tag = line[pipe_first + 1 : pipe_last]
                new_tag = convert_tag_upper(tag)
                new_line = line[: pipe_first + 1] + new_tag + line[pipe_last:]

                print(new_line.strip())

    else:
        print("File not found.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
