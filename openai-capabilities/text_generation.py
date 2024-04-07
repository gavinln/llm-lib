"""
https://platform.openai.com/docs/guides/embeddings
"""
import logging
import pathlib
from typing import Any, Iterable

import fire
import openai
import tiktoken
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import \
    ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)


def get_chat_completion(response_format: str) -> ChatCompletion:
    content = """
        The Los Angeles Dodgers won the World Series in 2020.
    """
    json_rf: ResponseFormat = {"type": "json_object"}
    rf = json_rf if response_format == "json" else openai.NOT_GIVEN
    completion = OpenAI().chat.completions.create(
        response_format=rf,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant designed to output JSON.",
            },
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": content,
            },
            {"role": "user", "content": "Where was it played?"},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    return completion


def chat_completion_text():
    "complete text input with text output"

    completion = get_chat_completion("")

    print(completion.choices[0].message.content)

    json_format = completion.model_dump_json(indent=2)
    print(json_format)


def chat_completion_json():
    "complete text input with json output"

    completion = get_chat_completion("json")

    print(completion.choices[0].message.content)

    json_format = completion.model_dump_json(indent=2)
    print(json_format)


def get_messages() -> Iterable[ChatCompletionMessageParam]:
    msg1: ChatCompletionMessageParam = {
        "role": "system",
        "content": (
            "You are a helpful, pattern-following assistant that "
            "translates corporate jargon into plain English."
        ),
    }
    msg2: ChatCompletionMessageParam = {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    }
    msg3: ChatCompletionMessageParam = {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    }
    msg4: ChatCompletionMessageParam = {
        "role": "system",
        "name": "example_user",
        "content": (
            "Let's circle back when we have more bandwidth to touch "
            "base on opportunities for increased leverage."
        ),
    }
    msg5: ChatCompletionMessageParam = {
        "role": "system",
        "name": "example_assistant",
        "content": (
            "Let's talk later when we're less busy about" "how to do better."
        ),
    }
    msg6: ChatCompletionMessageParam = {
        "role": "user",
        "content": (
            "This late pivot means we don't have time to boil the "
            "ocean for the client deliverable."
        ),
    }
    return [msg1, msg2, msg3, msg4, msg5, msg6]


def num_tokens_from_messages(messages):
    "Returns the number of tokens used by a list of messages."
    model = "gpt-3.5-turbo-0613"
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                # if there's a name, the role is omitted
                if key == "name":
                    num_tokens += -1
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"not presently implemented for model {model}."
        )


def count_tokens_local():
    "count tokens using the tiktoken library"

    messages = get_messages()
    for message in messages:
        print(message)
    token_count = num_tokens_from_messages(messages)
    print(f"token count = {token_count}")


def count_tokens_remote():
    "count tokens using an openai call"

    messages = get_messages()
    model = "gpt-3.5-turbo-0613"

    completion: ChatCompletion = OpenAI().chat.completions.create(
        messages=messages, model=model, temperature=0
    )
    json_format = completion.model_dump_json(indent=2)
    print(json_format)

    if completion.usage:
        print(f"token count = {completion.usage.prompt_tokens}")


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "chat-completion-text": chat_completion_text,
            "chat-completion-json": chat_completion_json,
            "count-tokens-local": count_tokens_local,
            "count-tokens-remote": count_tokens_remote,
        }
    )


if __name__ == "__main__":
    main()
