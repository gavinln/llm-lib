"""
https://cookbook.openai.com/examples/how_to_use_guardrails
"""

import logging
import math
import statistics
import sys
import textwrap
import typing
from pprint import pprint as pp

import fire
import numpy as np
import openai

GPT_MODEL = "gpt-4o"
# GPT_MODEL = "gpt-4o-mini"


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
    "get multiple log probabilities"
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


def get_chat_response_prob(user_request):
    "get log probabilities, one for each token"
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_request},
    ]
    completion = get_completion(messages, logprobs=True)
    lps = []
    for content in completion.choices[0].logprobs.content:
        lp = LogProbs(
            content.token, content.logprob, logprob_to_prob(content.logprob)
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


ada_lovelace_article = """
Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27
November 1852) was an English mathematician and writer, chiefly known for her
work on Charles Babbage's proposed mechanical general-purpose computer, the
Analytical Engine. She was the first to recognise that the machine had
applications beyond pure calculation.

Ada Byron was the only legitimate child of poet Lord Byron and reformer Lady
Byron. All Lovelace's half-siblings, Lord Byron's other children, were born out
of wedlock to other women. Byron separated from his wife a month after Ada was
born and left England forever. He died in Greece when Ada was eight. Her mother
was anxious about her upbringing and promoted Ada's interest in mathematics and
logic in an effort to prevent her from developing her father's perceived
insanity. Despite this, Ada remained interested in him, naming her two sons
Byron and Gordon. Upon her death, she was buried next to him at her request.
Although often ill in her childhood, Ada pursued her studies assiduously. She
married William King in 1835. King was made Earl of Lovelace in 1838, Ada
thereby becoming Countess of Lovelace.

Her educational and social exploits brought her into contact with scientists
such as Andrew Crosse, Charles Babbage, Sir David Brewster, Charles Wheatstone,
Michael Faraday, and the author Charles Dickens, contacts which she used to
further her education. Ada described her approach as "poetical science" and
herself as an "Analyst (& Metaphysician)".

When she was eighteen, her mathematical talents led her to a long working
relationship and friendship with fellow British mathematician Charles Babbage,
who is known as "the father of computers". She was in particular interested in
Babbage's work on the Analytical Engine. Lovelace first met him in June 1833,
through their mutual friend, and her private tutor, Mary Somerville.

Between 1842 and 1843, Ada translated an article by the military engineer Luigi
Menabrea (later Prime Minister of Italy) about the Analytical Engine,
supplementing it with an elaborate set of seven notes, simply called "Notes".

Lovelace's notes are important in the early history of computers, especially
since the seventh one contained what many consider to be the first computer
program—that is, an algorithm designed to be carried out by a machine. Other
historians reject this perspective and point out that Babbage's personal notes
from the years 1836/1837 contain the first programs for the engine. She also
developed a vision of the capability of computers to go beyond mere calculating
or number-crunching, while many others, including Babbage himself, focused only
on those capabilities. Her mindset of "poetical science" led her to ask
questions about the Analytical Engine (as shown in her notes) examining how
individuals and society relate to technology as a collaborative tool.
"""

hallucination_prompt = """
You retrieved this article: {article}. The question is: {question}.

Before even answering the question, consider whether you have sufficient
information in the article to answer the question fully.

Your output should JUST be the boolean true or false, of if you have sufficient
information in the article to answer the question.

Respond with just one word, the boolean true or false. You must output the word
'True', or the word 'False', nothing else.
"""


def print_token_logprobs(article, questions):
    for question in questions:
        print(question)
        lps = get_chat_response_probs(
            hallucination_prompt.format(article=article, question=question),
            top_logprobs=1,
        )
        print(textwrap.indent("\n".join(str(lp) for lp in lps), "\t"))


def reduce_hallucinations():
    # Questions that can be easily answered given the article
    easy_questions = [
        "What nationality was Ada Lovelace?",
        "What was an important finding from Lovelace's seventh note?",
    ]

    # Questions that are not fully covered in the article
    medium_questions = [
        "Did Lovelace collaborate with Charles Dickens",
        "What concepts did Lovelace build with Charles Babbage",
    ]

    print_token_logprobs(ada_lovelace_article, easy_questions)
    print_token_logprobs(ada_lovelace_article, medium_questions)


sentence_list = [
    "My",
    "My least",
    "My least favorite",
    "My least favorite TV",
    "My least favorite TV show",
    "My least favorite TV show is",
    "My least favorite TV show is Breaking Bad",
]


def autocomplete():
    complete_sentence_prompt = """
    Complete this sentence. You are acting as auto-complete. Simply complete
    the sentence to the best of your ability, make sure it is just ONE
    sentence: {sentence}
    """

    high_prob_completions = {}
    low_prob_completions = {}
    for sentence in sentence_list:
        print(sentence)
        lps = get_chat_response_probs(
            complete_sentence_prompt.format(sentence=sentence), top_logprobs=3
        )
        print("\t" + ", ".join(f"{lp.token}: {lp.prob}" for lp in lps))
        for lp in lps:
            if lp.prob > 95:
                high_prob_completions[sentence] = lp.token
            if lp.prob < 60:
                low_prob_completions[sentence] = lp.token

    print("high prob completions", "=" * 10)
    pp(high_prob_completions)
    print("low prob completions", "=" * 10)
    pp(low_prob_completions)


def perplexity():
    sentences = [
        "In a short sentence, has artifical intelligence "
        + "grown in the last decade?",
        "In a short sentence, what are your thoughts on "
        + "the future of artificial intelligence?",
    ]

    for user_request in sentences:
        lps = get_chat_response_prob(user_request)
        print("prompt")
        print("    " + user_request)
        logprobs = []
        for token, logprob, prob in lps:
            print("    " + token.strip(), round(logprob, 2))
            logprobs.append(logprob)
        perplexity_score = math.exp(-statistics.mean(logprobs))
        print(f"{perplexity_score=:.2f}")


def main():
    print(sys.version)
    fire.Fire(
        {
            "classification": classification,
            "reduce-hallucinations": reduce_hallucinations,
            "autocomplete": autocomplete,
            "perplexity": perplexity,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
