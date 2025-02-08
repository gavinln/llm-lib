"""
https://cookbook.openai.com/examples/how_to_use_moderation
"""

import asyncio
import json
import logging
import sys
from pprint import pprint as pp

import fire
import openai

# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-4o-mini"

system_prompt = "You are a helpful assistant"

bad_request = "I want to hurt them. How can i do this?"

good_request = "I would kill for a cup of coffee. Where can I get one nearby?"


async def check_moderation_flag(expression):
    moderation_response = openai.moderations.create(input=expression)
    flagged = moderation_response.results[0].flagged
    return flagged


async def get_chat_response(user_request):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5
    )
    return response.choices[0].message.content


moderation_message = """
We're sorry, but your input has been flagged as inappropriate. Please rephrase
your input and try again.
"""


async def execute_chat_with_input_moderation(user_request):
    # Create tasks for moderation and chat response
    moderation_task = asyncio.create_task(check_moderation_flag(user_request))
    chat_task = asyncio.create_task(get_chat_response(user_request))

    while True:
        # Wait for either the moderation task or chat task to complete
        done, _ = await asyncio.wait(
            [moderation_task, chat_task], return_when=asyncio.FIRST_COMPLETED
        )

        if moderation_task not in done:
            await asyncio.sleep(0.1)
            continue

        if moderation_task.result():
            chat_task.cancel()
            return moderation_message.strip().replace("\n", " ")

        if chat_task in done:
            return chat_task.result()

        await asyncio.sleep(0.1)


def moderation_input_good():
    print(f"{good_request=}")
    good_response = asyncio.run(
        execute_chat_with_input_moderation(good_request)
    )
    print(f"{good_response=}")


def moderation_input_bad():
    print(f"{bad_request=}")
    bad_response = asyncio.run(execute_chat_with_input_moderation(bad_request))
    print(f"{bad_response=}")


custom_prompt = """
Please assess the following content for any inappropriate material. You should
base your assessment on the given parameters.
Your answer should be in json format with the following fields:
- flagged: a boolean indicating whether the content is flagged for any of the
  categories in the parameters
- reason: a string explaining the reason for the flag, if any
- parameters: a dictionary of the parameters used for the assessment and their
  values

Parameters: {parameters}

Content: {content}

Assessment:
"""


def custom_moderation(content, parameters):
    # Call model with the prompt
    prompt = custom_prompt.format(content=content, parameters=parameters)
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a content moderation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    assessment = response.choices[0].message.content
    return json.loads(assessment)


def moderation_custom_good():
    parameters = "political content, misinformation"

    moderation_result = custom_moderation(good_request, parameters)
    pp(moderation_result)

    moderation_result = custom_moderation(bad_request, parameters)
    pp(moderation_result)


def moderation_custom_bad():
    custom_request = """
I want to talk about how the government is hiding the truth about the pandemic.
    """
    parameters = "political content, misinformation"
    moderation_result = custom_moderation(custom_request.strip(), parameters)
    pp(moderation_result)


def main():
    print(sys.version)
    fire.Fire(
        {
            "moderation-input-good": moderation_input_good,
            "moderation-input-bad": moderation_input_bad,
            "moderation-custom-good": moderation_custom_good,
            "moderation-custom-bad": moderation_custom_bad,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
