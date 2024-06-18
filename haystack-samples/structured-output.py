"""
https://haystack.deepset.ai/tutorials/28_structured_output_with_loop

OPENAI_API_KEY needed
"""

import json
import logging
import pathlib
import sys
import textwrap
from pprint import pprint
from typing import Type

from colorama import Fore, Style
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from pydantic import BaseModel, ValidationError

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: list[City]


def get_prompt_template() -> str:
    prompt_template = """

    Create a JSON object from the information present in this passage:
    {{passage}}.

    Only use information that is present in the passage. Follow
    this JSON schema, but only return the actual instances without any
    additional schema definition:

    {{schema}}

    Make sure your response is a dict and not a list.
    {% if invalid_replies and error_message %}

        You already created the following output in a previous attempt:
        {{invalid_replies}}

        However, this doesn't comply with the format requirements from above
        and triggered this Python exception: {{error_message}}

        Correct the output and try again. Just return the corrected output
        without any extra explanations.

    {% endif %}
    """
    return prompt_template


@component
class OutputValidator:
    def __init__(self, pydantic_model: Type[BaseModel]):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    # Define the component output
    @component.output_types(
        valid_replies=list[str],
        invalid_replies=list[str] | None,
        error_message=str | None,
    )
    def run(self, replies: list[str]):

        self.iteration_counter += 1

        # If the LLM's reply is a valid object, return `"valid_replies"`
        try:
            output_dict = json.loads(replies[0])
            self.pydantic_model.model_validate(output_dict)
            print(
                Fore.GREEN
                + f"OutputValidator at Iteration {self.iteration_counter}: "
                + "Valid JSON from LLM - No need for looping:"
                # + f"Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            print(Style.RESET_ALL)
            return {"valid_replies": replies}

        # If the LLM's reply is corrupted or not valid,
        # return "invalid_replies" and the "error_message"
        # for LLM to try again
        except (ValueError, ValidationError) as e:
            print(
                Fore.RED
                + f"OutputValidator at Iteration {self.iteration_counter}: "
                + "Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0]} \n"
                f"Error from OutputValidator: {e}"
            )
            print(Style.RESET_ALL)
            return {"invalid_replies": replies, "error_message": str(e)}


def print_validator_replies(result):
    if "output_validator" in result:
        validator = result["output_validator"]
        if "valid_replies" in validator:
            valid_replies = validator["valid_replies"]
            for reply in valid_replies:
                print(reply)


def main():
    json_schema = CitiesData.model_json_schema()
    pprint(json_schema)
    prompt_template = textwrap.dedent(get_prompt_template())
    prompt_builder = PromptBuilder(template=prompt_template)

    generator = OpenAIGenerator()

    output_validator = OutputValidator(pydantic_model=CitiesData)

    pipeline = Pipeline(max_loops_allowed=5)

    # Add components to your pipeline
    pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    pipeline.add_component(instance=generator, name="llm")
    pipeline.add_component(instance=output_validator, name="output_validator")

    # Now, connect the components to each other
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm", "output_validator")

    # If a component has more than one output or input,
    # explicitly specify the connections:
    pipeline.connect(
        "output_validator.invalid_replies", "prompt_builder.invalid_replies"
    )
    pipeline.connect(
        "output_validator.error_message", "prompt_builder.error_message"
    )

    passage = """
    Berlin is the capital of Germany. It has a population of 3,850,809. Paris,
    France's capital, has 2.161 million residents. Lisbon is the capital and
    the largest city of Portugal with the population of 504,718.
    """
    result = pipeline.run(
        {"prompt_builder": {"passage": passage, "schema": json_schema}}
    )
    print_validator_replies(result)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
