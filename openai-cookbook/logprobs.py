"""
https://cookbook.openai.com/examples/how_to_use_guardrails
"""

import logging
import sys
import typing
import textwrap

import numpy as np
import fire
import openai

# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-4o-mini"


def logprob_to_prob(logprob: float):
    return float(np.round(np.exp(logprob) * 100))


def get_completion(
    messages: list[dict[str, str]],
    model: str = GPT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0,
    stop=None,
    seed: int = 123,
    tools=None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = openai.chat.completions.create(**params)
    return completion


classification_request = """
You will be given a headline of a news article. Classify the article into one
of the following categories: Technology, Politics, Sports, Art.
Return only the name of the category, and nothing else. MAKE SURE your output
is one of the four categories stated.
Article headline: {}
"""


def get_chat_response(user_request):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_request},
    ]
    completion = get_completion(messages)
    response = completion.choices[0].message.content
    return response


class LogProbs(typing.NamedTuple):
    token: str
    logprob: float
    prob: float


def get_chat_response_probs(user_request, top_logprobs=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_request},
    ]
    if top_logprobs:
        completion = get_completion(
            messages, logprobs=True, top_logprobs=top_logprobs
        )
    else:
        completion = get_completion(messages)

    lps = []
    for logprob in completion.choices[0].logprobs.content[0].top_logprobs:
        lp = LogProbs(
            logprob.token, logprob.logprob, logprob_to_prob(logprob.logprob)
        )
        lps.append(lp)
    return lps


def classification():
    headlines = [
        "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
        "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
        "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
    ]
    for headline in headlines:
        print(headline)
        response = get_chat_response(classification_request.format(headline))
        print(f"{response=}")

    for headline in headlines:
        print(headline)
        lps = get_chat_response_probs(
            classification_request.format(headline), top_logprobs=2
        )
        print(textwrap.indent("\n".join(str(lp) for lp in lps), "\t"))


def main():
    print(sys.version)
    fire.Fire(
        {
            "classification": classification,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
