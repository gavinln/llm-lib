"""
https://cookbook.openai.com/examples/entity_extraction_for_long_documents
"""

import logging
import pathlib
import tempfile
from typing import Any

import openai
import textract
import tiktoken
from joblib import Memory

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"


def encode_text(text, encoding_name=EMBEDDING_ENCODING):
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


def get_financial_pdf():
    pdf_file = (
        SCRIPT_DIR
        / "data"
        / "fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf"
    )
    assert pdf_file.exists(), "Cannot find file {}".format(pdf_file)
    return pdf_file


def get_financial_pdf_text(pdf_file: pathlib.Path) -> str:
    text = textract.process(pdf_file, method="pdfminer").decode("utf-8")
    clean_text = text.replace("  ", " ").replace("\n", "; ").replace(";", " ")
    return clean_text


def simple_entity_extraction():

    document = "<document>"

    template_prompt = f'''

    Extract key pieces of information from this regulation document.
    If a piece of information is not present, output \"NOT SPECIFIED\".
    When you extract a piece of information, include the closest page number.

    Use the following format:

    0. Who is the author
    1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR
    2. What is the value of External Manufacturing Costs in USD
    3. What is the Capital Expenditure Limit in USD

    Document: """{document}"""

    0. Who is the author: Tom Anderson (Page 1)
    '''
    return template_prompt


def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a
        # range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def complete_chat(prompt) -> str:
    messages: Any = [
        {
            "role": "system",
            # "content": "You help extract information from documents.",
            "content": prompt,
        },
        {"role": "user", "content": prompt},
    ]

    response = openai.OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    assert len(response.choices) > 0, "no choices available"
    assert response.choices[0].message.content, "mesage content missing"
    return response.choices[0].message.content


@memory.cache
def create_text_chunks(document: str) -> list[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    enc_chunks = create_chunks(document, 1000, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in enc_chunks]
    return text_chunks


def main():
    # get document and prompt
    document = get_financial_pdf_text(get_financial_pdf())
    sse_prompt = simple_entity_extraction()
    print(sse_prompt)

    text_chunks: Any = create_text_chunks(document)

    for chunk in text_chunks:
        prompt = sse_prompt.replace("<document>", chunk)
        result = complete_chat(prompt)
        print(result)
        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
