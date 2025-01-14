"""
Named Entity Recognition to Enrich Text|CompletionsFunctions|Oct 20, 2023
https://cookbook.openai.com/examples/named_entity_recognition_to_enrich_text

Uses the https://pypi.org/project/nlpia2-wikipedia/ library which is a thin
Python wrapper around Wikipedia

Also displays prompt and completion tokens count
"""

import json
import logging
import pathlib
from typing import Any, Optional

import openai
import wikipedia
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_random_exponential


from rich.console import Console
from rich.markdown import Markdown

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

GPT_MODEL = "gpt-4-turbo"


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
)
def chat_completion_request(
    messages, tools: Any = None, tool_choice: Any = None, model=GPT_MODEL
) -> Optional[ChatCompletion]:
    try:
        response: ChatCompletion = openai.OpenAI().chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
    return None


def get_labels() -> list[str]:
    """
    Returns a list of predefined named entity labels used for text enrichment.

    These labels represent different categories of named entities that can be
    identified within a text. They include types such as:

    Returns:
        list[str]: A list of strings where each string is a named entity label.
    """
    labels: list[str] = [
        "person",  # people, including fictional characters
        "fac",  # buildings, airports, highways, bridges
        "org",  # organizations, companies, agencies, institutions
        "gpe",  # geopolitical entities like countries, cities, states
        "loc",  # non-gpe locations
        "product",  # vehicles, foods, appareal, appliances, software, toys
        "event",  # named sports, scientific milestones, historical events
        "work_of_art",  # titles of books, songs, movies
        "law",  # named laws, acts, or legislations
        "language",  # any named language
        "date",  # absolute or relative dates or periods
        "time",  # time units smaller than a day
        "percent",  # percentage (e.g., "twenty percent", "18%")
        "money",  # monetary values, including unit
        "quantity",  # measurements, e.g., weight or distance
    ]
    return labels


def system_message(labels):
    return f"""
        You are an expert in Natural Language Processing. Your task is
        to identify common Named Entities (NER) in a given text.
        The possible common Named Entities (NER) types are exclusively:
        ({", ".join(labels)}).
    """


def assisstant_message():
    return f"""
    EXAMPLE:
        Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg
        invented the movable-type printing press. His work led to an
        information revolution and the unprecedented mass-spread of
        literature throughout Europe. Modelled on the design of the
        existing screw presses, a single Renaissance movable-type
        printing press could produce up to 3,600 pages per workday.'
        {{
            "gpe": ["Germany", "Europe"],
            "date": ["1440"],
            "person": ["Johannes Gutenberg"],
            "product": ["movable-type printing press"],
            "event": ["Renaissance"],
            "quantity": ["3,600 pages"],
            "time": ["workday"]
        }}
    """


def user_message(text):
    return f"""
    TASK:
        Text: {text}
    """


class Messages(list):
    def add_system(self, content: str):
        print("---system--- " + content)
        self.append({"role": "system", "content": content})

    def add_user(self, content: str):
        print("---user--- " + content)
        self.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        print("---assistant--- " + content)
        self.append({"role": "assistant", "content": content})

    def add(self, content: dict):
        self.append(content)


def wikipedia_example():
    titles = wikipedia.search("New York")
    print(f"{len(titles)=}")
    page = wikipedia.page(titles[1])
    print(f"{page.title=}")
    print(f"{page.url=}")
    print(f"{page.content[:25]=}")
    print(f"{page.summary[:25]=}")
    print(f"{len(page.links)=}")


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
def find_link(entity: str) -> Optional[str]:
    "Finds a Wikipedia link for a given entity"
    try:
        titles = wikipedia.search(entity)
        if titles:
            # naively consider the first result as the best
            page = wikipedia.page(titles[0])
            return page.url
    except wikipedia.exceptions.DisambiguationError as ex:
        log.debug(str(ex))
    except wikipedia.exceptions.WikipediaException as ex:
        log.debug(
            "Error while searching for Wikipedia link for entity {}: {}".format(
                entity, str(ex)
            )
        )

    return None


def test_find_link():
    link = find_link("English")
    assert True


def find_all_links(label_entities: dict) -> dict:
    """
    Finds all Wikipedia links for the dictionary entities in the whitelist label list.
    """
    whitelist = ["event", "gpe", "org", "person", "product", "work_of_art"]

    link_dict = {
        e: find_link(e)
        for label, entities in label_entities.items()
        for e in entities
        if label in whitelist
    }
    return {key: val for key, val in link_dict.items() if not val is None}


def get_example_label_entities():
    label_entities = {
        "gpe": ["Germany"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["printing press"],
    }
    return label_entities


def test_find_all_links():
    text_with_links = find_all_links(get_example_label_entities())
    print(text_with_links)


def enrich_entities(text: str, label_entities: dict) -> str:
    entity_link_dict = find_all_links(label_entities)
    print(f"{entity_link_dict=}")
    for entity, link in entity_link_dict.items():
        text = text.replace(entity, f"[{entity}]({link})")
    return text


def test_enrich_entities():
    text = """
        In Germany, in 1440, Johannes Gutenberg invented the printing press
    """
    label_entities = get_example_label_entities()


def generate_functions(labels: list[str]) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "enrich_entities",
                "description": "Enrich Text with Knowledge Base Links",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "r'^("
                        + f"{'|'.join(labels)}"
                        + ")$'": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]


def print_token_counts(response: ChatCompletion):
    if response.usage:
        console = Console()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        console.print(f"{prompt_tokens=}, {completion_tokens=}")


def run_enrich_text_task(labels: list[str], text: str):
    tools = generate_functions(labels)
    print(tools)

    messages = Messages()
    messages.add_system(system_message(labels))
    messages.add_assistant(assisstant_message())
    messages.add_user(user_message(text))

    tool_choice = {"type": "function", "function": {"name": "enrich_entities"}}
    response: Optional[ChatCompletion] = chat_completion_request(
        messages, tools=tools, tool_choice=tool_choice
    )
    if response is None or response.choices is None:
        return None

    print_token_counts(response)

    message: ChatCompletionMessage = response.choices[0].message
    if message is None:
        return None

    print(f"{message=}")
    if message.tool_calls is None:
        return None

    available_functions = {"enrich_entities": enrich_entities}
    if len(message.tool_calls) > 0:
        function_name = message.tool_calls[0].function.name

        function_to_call = available_functions[function_name]
        print(f"{function_to_call=}")

        function_args = json.loads(message.tool_calls[0].function.arguments)
        print(f"{function_args=}")
        function_response = function_to_call(text, function_args)

        return function_response

    return None


def print_markdown(md_text: str):
    console = Console()
    md = Markdown(md_text)
    console.print(md)


def enrich_text_task():
    text = """
        The Beatles were an English rock band formed in Liverpool in 1960,
        comprising John Lennon, Paul McCartney, George Harrison,
        and Ringo Starr.
    """
    labels = get_labels()
    # test_find_all_links()
    result = run_enrich_text_task(labels, text)
    if result:
        print_markdown(result)


def main():
    logging.basicConfig(level=logging.INFO)
    print("In file", pathlib.Path(globals()["__file__"]).name)
    # wikipedia_example()
    # test_find_link()
    enrich_text_task()


if __name__ == "__main__":
    main()
