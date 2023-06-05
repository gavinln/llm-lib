import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


# basic example
def basic_example(prompt):
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=7
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def question_answer(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def correct_grammar(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Correct this to standard English:\n\nShe no went to the market.",
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def summarize_second_grader(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def translate_english(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def sql_translate(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def parse_unstructured_data(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def classification(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def python_to_natural_language(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def time_complexity_of_code(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def translate_programming_languages(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"]
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))

