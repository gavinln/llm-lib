"""

Extracting structured output
https://python.langchain.com/v0.1/docs/use_cases/extraction/

"""

import logging
import pathlib
import tempfile
from typing import Optional

import fire
from joblib import Memory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


memory = Memory(tempfile.gettempdir(), verbose=0)


class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(
        default=None, description="The name of the person"
    )
    hair_color: Optional[str] = Field(
        default=None, description="The color of the peron's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


def get_extraction_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    return prompt


def structured_one_entity():
    prompt = get_extraction_prompt()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=Person)
    text = "Alan Smith is 6 feet tall and has blond hair."
    print(f"{text=}")
    out = runnable.invoke({"text": text})
    print(type(out))


class Data(BaseModel):
    """Extracted data about people."""

    people: list[Person]


def structured_many_entities():
    prompt = get_extraction_prompt()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=Data)
    text = (
        "My name is Jeff, my hair is black and i am 6 feet tall. Anna "
        "has the same color hair as me."
    )
    print(f"{text=}")
    out = runnable.invoke({"text": text})
    print(type(out))
    print(out)


def main():
    fire.Fire(
        {
            "structured-one-entity": structured_one_entity,
            "structured-many-entities": structured_many_entities,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
