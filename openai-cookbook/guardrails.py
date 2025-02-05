"""
https://cookbook.openai.com/examples/how_to_use_guardrails
"""

import asyncio
import logging
import sys
from asyncio import create_task, wait

import fire
import openai

GPT_MODEL = "gpt-4o-mini"


"""
# Input guardrails

Topical guardrails: Identify when a user asks an off-topic question and give
them advice on what topics the LLM can help them with.

Jailbreaking: Detect when a user is trying to hijack the LLM and override its
prompting.

Prompt injection: Pick up instances of prompt injection where users try to hide
malicious code that will be executed in any downstream functions the LLM
executes.
"""

guardrail_content = """
Your role is to assess whether the user question is allowed or not. The allowed
topics are cats and dogs. If the topic is allowed, say 'allowed' otherwise say
'not_allowed'
"""


async def get_chat_response(user_request):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5
    )
    return response.choices[0].message.content


async def topical_guardrail(user_request):
    messages = [
        {
            "role": "system",
            "content": guardrail_content,
        },
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0
    )
    return response.choices[0].message.content


async def execute_chat_with_guardrail(user_request):
    chat_response_task = create_task(get_chat_response(user_request))
    topical_guardrail_task = create_task(topical_guardrail(user_request))
    while True:
        done, _ = await wait(
            [chat_response_task, topical_guardrail_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if topical_guardrail_task in done:
            guardrail_response = topical_guardrail_task.result()
            if guardrail_response == "not_allowed":
                chat_response_task.cancel()
                return "I can only talk about cats and dogs"
            elif chat_response_task in done:
                return chat_response_task.result()
        else:
            await asyncio.sleep(0.1)


def input_guardrails(user_request):
    response = asyncio.run(execute_chat_with_guardrail(user_request))
    print("response ----------")
    print(response)


def input_good():
    good_request = "What are the best breeds of dog for people that like cats?"
    input_guardrails(good_request)


def input_bad():
    bad_request = "I want to talk about horses"
    input_guardrails(bad_request)


domain = "animal breed recommendation"

animal_advice_criteria = """
Assess the presence of explicit recommendation of cat or dog breeds in the content.
The content should contain only general advice about cats and dogs, not specific breeds to purchase."""

animal_advice_steps = """
1. Read the content and the criteria carefully.
2. Assess how much explicit recommendation of cat or dog breeds is contained in the content.
3. Assign an animal advice score from 1 to 5, with 1 being no explicit cat or dog breed advice, and 5 being multiple named cat or dog breeds.
"""

moderation_system_prompt = """
You are a moderation assistant. Your role is to detect content about {domain}
in the text provided, and mark the severity of that content.

## {domain}

### Criteria

{scoring_criteria}

### Instructions

{scoring_steps}

### Content

{content}

### Evaluation (score only!)
"""

moderation_message = """
Sorry, we're not permitted to give animal breed advice. I can help you with any
general queries you might have.
"""


async def moderation_guardrail(chat_response: str):
    mod_messages = [
        {
            "role": "user",
            "content": moderation_system_prompt.format(
                domain=domain,
                scoring_criteria=animal_advice_criteria,
                scoring_steps=animal_advice_steps,
                content=chat_response,
            ),
        },
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=mod_messages, temperature=0
    )
    return response.choices[0].message.content


def contained_str_count(text: str, match: str):
    "print number of times match contained in text"
    flag_str = match.lower()
    flag_str_count = text.lower().count(flag_str)
    print(f"{flag_str} contained in response {flag_str_count} time(s)")


async def execute_all_guardrails(user_request):
    chat_response_task = create_task(get_chat_response(user_request))
    topical_guardrail_task = create_task(topical_guardrail(user_request))
    while True:
        done, _ = await wait(
            [chat_response_task, topical_guardrail_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if topical_guardrail_task in done:
            guardrail_response = topical_guardrail_task.result()
            if guardrail_response == "not_allowed":
                chat_response_task.cancel()
                return "I can only talk about cats and dogs"
            elif chat_response_task in done:
                chat_response = chat_response_task.result()
                moderation_response = await moderation_guardrail(chat_response)
                if int(moderation_response) >= 3:
                    print("moderation tripped ----------")
                    print(chat_response)
                    return moderation_message.strip()
                else:
                    return chat_response

        else:
            await asyncio.sleep(0.1)


def output_guardrails():
    print("request 1 ==========")
    user_request = "What is some advice you can give to a new dog owner?"
    response = asyncio.run(execute_all_guardrails(user_request))
    print(user_request)
    print("response ----------")
    print(response)

    print("request 2 ==========")
    user_request = "What are the best breeds of dog for people that like cats?"
    response = asyncio.run(execute_all_guardrails(user_request))
    print(user_request)
    print("response ----------")
    print(response)


def main():
    print(sys.version)
    fire.Fire(
        {
            "input-good": input_good,
            "input-bad": input_bad,
            "output-guardrails": output_guardrails,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
