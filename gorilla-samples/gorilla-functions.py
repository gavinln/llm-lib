"""
https://github.com/ShishirPatil/gorilla/tree/main/openfunctions/openfunctions-v1

https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html
"""

import json
import logging
import pathlib
import pprint as pp

import fire
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def get_gorilla_response(
    query,
    model,
    functions=[],
):
    openai.api_key = "EMPTY"
    openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": query}],
            functions=functions,
        )
        return completion.choices[0].message.content  # type: ignore
    except Exception as e:
        print(e, model, query)


def get_uber_ride_function() -> dict:
    return {
        "name": "Uber Carpool",
        "api_name": "uber.ride",
        "description": "Find suitable ride for customers given the "
        + "location, type of ride, and the amount of time the customer "
        + "is willing to wait as parameters",
        "parameters": [
            {
                "name": "loc",
                "description": "location of the starting place of the "
                + "uber ride",
            },
            {
                "name": "type",
                "enum": ["plus", "comfort", "black"],
                "description": "types of uber ride user is ordering",
            },
            {
                "name": "time",
                "description": "the amount of time in minutes the "
                + "customer is willing to wait",
            },
        ],
    }


def get_current_weather_function() -> dict:
    return {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }


def get_recursive_size(root_dir) -> int:
    root_path = pathlib.Path(root_dir)
    return sum(file.stat().st_size for file in root_path.rglob("*"))


def get_hub_model_sizes(cache_dir):
    cache_path = pathlib.Path(cache_dir)
    hub_dir = cache_path / "hub"
    hub_models = [f for f in hub_dir.glob("model*") if f.is_dir()]
    return {model: get_recursive_size(model) for model in hub_models}


def show_huggingface_cached_models():
    "show huggingface cached models"
    cache_dir = pathlib.Path("~") / ".cache" / "huggingface"
    cache_dir = cache_dir.expanduser().resolve()
    model_size_dict = get_hub_model_sizes(cache_dir)
    size_name_dict = {
        f"{size:15,d}": p.name for p, size in model_size_dict.items()
    }
    print(
        "\n".join(size + " " + name for size, name in size_name_dict.items())
    )
    return ""


def hosted_functions_v1():
    "run gorilla function call on hosted model v1"
    model = "gorilla-openfunctions-v1"
    query = (
        'Call me an Uber ride type "Plus" in Berkeley at zipcode '
        + "94704 in 10 minutes"
    )
    functions = [get_uber_ride_function()]
    out = get_gorilla_response(query, model, functions=functions)
    print(out)


def get_prompt(user_query: str, functions: list = []) -> str:
    """
    Generates a conversation prompt based on the user's query and a list of functions.

    Parameters:
    - user_query (str): The user's query.
    - functions (list): A list of functions to include in the prompt.

    Returns:
    - str: The formatted conversation prompt.
    """
    if len(functions) == 0:
        return f"USER: <<question>> {user_query}\nASSISTANT: "
    functions_string = json.dumps(functions)
    return f"USER: <<question>> {user_query} <<function>> {functions_string}\nASSISTANT: "


def local_functions():
    "run gorilla function call on local torch model"
    # Device setup
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model and tokenizer setup
    model_id: str = "gorilla-llm/gorilla-openfunctions-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    # Move model to device
    model.to(device)

    # Pipeline setup
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Example usage
    query: str = (
        'Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes'
    )
    functions = [
        {
            "name": "Uber Carpool",
            "api_name": "uber.ride",
            "description": "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
            "parameters": [
                {
                    "name": "loc",
                    "description": "Location of the starting place of the Uber ride",
                },
                {
                    "name": "type",
                    "enum": ["plus", "comfort", "black"],
                    "description": "Types of Uber ride user is ordering",
                },
                {
                    "name": "time",
                    "description": "The amount of time in minutes the customer is willing to wait",
                },
            ],
        }
    ]
    functions = [get_uber_ride_function()]

    # Generate prompt and obtain model output
    prompt = get_prompt(query, functions=functions)
    output = pipe(prompt)
    pp.pprint(output)


def hosted_functions_v2():
    "run gorilla function call on hosted model v2"
    model = "gorilla-openfunctions-v2"
    query = "What's the weather like in the two cities of Boston and San Francisco?"
    functions = [get_current_weather_function()]
    out = get_gorilla_response(query, model, functions=functions)
    print(out)


def main():
    fire.Fire(
        {
            "hosted-functions-v1": hosted_functions_v1,
            "local-functions": local_functions,
            "hosted-functions-v2": hosted_functions_v2,
            "show-huggingface-cached-models": show_huggingface_cached_models,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
