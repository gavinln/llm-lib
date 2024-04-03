import logging
import os
import pathlib
from typing import Any

import openai
import pandas as pd
import tiktoken

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def remove_newlines(series):
    series = series.str.replace("\n", " ")
    series = series.str.replace("\\n", " ")
    series = series.str.replace("  ", " ")
    return series


def text_to_dataframe(text_dir):
    texts = []
    log.info(f"Searching dir {text_dir}")
    for idx, file in enumerate(text_dir.glob("*")):
        print(file, file.stem)
        text = file.read_text()
        texts.append((file.stem, text))
        # if idx > 3:
        #     break

    df = pd.DataFrame.from_records(texts, columns=["fname", "text"])
    df["text"] = df.fname + ". " + remove_newlines(df.text)
    return df


def print_token_count_distribution(n_tokens: pd.Series):
    bins = list(range(0, 10000, 500))
    tokens_bins: Any = pd.cut(n_tokens, bins)
    print(tokens_bins.value_counts())


def tokenize_scraped_text(scraped_file, tokenizer):
    df = pd.read_csv(scraped_file, index_col=0, names=["title", "text"])
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df


def split_into_many(text, tokenizer, max_tokens):
    """
    split the text into chunks of a maximum number of tokens
    """
    # Split the text into sentences
    sentences = text.split(". ")

    # Get the number of tokens for each sentence
    n_tokens = [
        len(tokenizer.encode(" " + sentence)) for sentence in sentences
    ]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in
        # the current sentence is greater than the max number of tokens,
        # then add the chunk to the list of chunks and reset the chunk
        # and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater
        # than the max number of tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number
        # of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def split_into_chunks_scraped_text(scraped_file, tokenizer, max_tokens):
    df = pd.read_csv(scraped_file, index_col=0, names=["title", "text"])
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    shortened = []

    for row in df.iterrows():
        # idx: Any = row[0]
        srs = row[1]
        if srs["text"] is None:
            continue

        # If the number of tokens is greater than the max number of tokens,
        # split the text into chunks
        if srs["n_tokens"] > max_tokens:
            shortened += split_into_many(srs["text"], tokenizer, max_tokens)
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(srs["text"])

    df2 = pd.DataFrame.from_records(
        [[item] for item in shortened], columns=["text"]
    )
    df2["n_tokens"] = df2.text.apply(lambda x: len(tokenizer.encode(x)))
    return df2


def create_embeddings(df, embedding_file):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    def get_embeddings(text):
        embedding_response: Any = client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return embedding_response.data[0].embedding

    df["embeddings"] = df.text.apply(lambda x: get_embeddings(x))
    df.to_csv(embedding_file)


def main():
    domain = "openai.com"
    processed_dir = pathlib.Path("processed")
    scraped_file = processed_dir / "scraped.csv"
    max_tokens = 500

    text_dir = pathlib.Path(f"text/{domain}")

    # Create a directory to store the csv files
    if not processed_dir.exists():
        os.mkdir("processed")

    if not scraped_file.exists():
        df = text_to_dataframe(text_dir)
        df.to_csv(scraped_file)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    df2: Any = tokenize_scraped_text(scraped_file, tokenizer)
    print_token_count_distribution(df2.n_tokens)

    df3: Any = split_into_chunks_scraped_text(
        scraped_file, tokenizer, max_tokens
    )
    print_token_count_distribution(df3.n_tokens)
    embedding_file = processed_dir / "embeddings.csv"
    create_embeddings(df3, embedding_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
