"""
https://cookbook.openai.com/examples/sdg1
"""

import logging
import pathlib
import re
import sys
import tempfile

import fire
import openai
import pandas as pd
from joblib import Memory

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)

# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-4o-mini"


def get_chat_response(user_request):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant designed to generate synthetic data.",
        },
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5
    )
    return response.choices[0].message.content


csv_prompt = """
Create a CSV file with 10 rows of housing data.
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms

Make sure that the numbers make sense (i.e. more rooms is usually bigger size,
more expensive locations increase price. more size is usually higher price etc.
make sure all the numbers make sense). Also only respond with the CSV.
"""


def csv_with_prompt():
    csv_data = get_chat_response(csv_prompt)
    print(csv_data)


csv_python = """
Create a Python program to generate 100 rows of housing data.
I want you to at the end of it output a pandas dataframe with 100 rows of data.
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms

Make sure that the numbers make sense (i.e. more rooms is usually bigger size,
more expensive locations increase price. more size is usually higher price etc.
make sure all the numbers make sense).
"""


def csv_with_python():
    csv_data_program = get_chat_response(csv_python)
    print(csv_data_program)


csv_multitable_python = """
Create a Python program to generate 3 different pandas dataframes.

1. Housing data
I want 100 rows. Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms
 - house type
 + any relevant foreign keys

2. Location
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - country
 - city
 - population
 - area (m^2)
 + any relevant foreign keys

3. House types
 - id (incrementing integer starting at 1)
 - house type
 - average house type price
 - number of houses
 + any relevant foreign keys

Make sure that the numbers make sense (i.e. more rooms is usually bigger size, more expensive locations increase price. more size is usually higher price etc. make sure all the numbers make sense).

Make sure that the dataframe generally follow common sense checks, e.g. the size of the dataframes make sense in comparison with one another.

Make sure the foreign keys match up and you can use previously generated dataframes when creating each consecutive dataframes.

You can use the previously generated dataframe to generate the next dataframe.
"""


def csv_multitable_with_python():
    csv_data_program = get_chat_response(csv_multitable_python)
    print(csv_data_program)


text_data_prompt = """
I am creating input output training pairs to fine tune my gpt model. The
usecase is a retailer generating a description for a product from a product
catalogue. I want the input to be product name and category (to which the
product belongs to) and output to be description.

The format should be of the form:
1.
Input: product_name, category
Output: description

2.
Input: product_name, category
Output: description

Do not add any extra characters around that formatting as it will make the
output parsing break. Create 5 training pairs.
"""


@memory.cache
def get_text_with_prompt():
    text_data = get_chat_response(text_data_prompt)
    return text_data


def text_with_prompt():
    pattern = re.compile(
        r"Input:\s*(.+?),\s*(.+?)\nOutput:\s*(.+?)(?=\n\n|\Z)", re.DOTALL
    )

    text_data = get_text_with_prompt()
    matches = pattern.findall(text_data)

    products = []
    categories = []

    for match in matches:
        product, category, description = match
        products.append(product)
        categories.append(category)

    df = pd.DataFrame({"product": products, "category": categories})
    print(df)


def text_imbalanced():
    print("NOT IMPLEMENTED")


def main():
    print(sys.version)
    fire.Fire(
        {
            "csv-with-prompt": csv_with_prompt,
            "csv-with-python": csv_with_python,
            "csv-multitable-with-python": csv_multitable_with_python,
            "text-with-prompt": text_with_prompt,
            "text-imbalanced": text_imbalanced,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
