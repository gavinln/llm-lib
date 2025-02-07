"""
https://cookbook.openai.com/examples/how_to_use_moderation
"""

import asyncio
import logging
import sys

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


def moderation_custom_good():
    pass


def moderation_custom_bad():
    pass


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
