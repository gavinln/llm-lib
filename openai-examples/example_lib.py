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
        presence_penalty=0.0,
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
        stop=["\n"],
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
        stop=["###"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def sentiment_classifier(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def explain_code(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['"""'],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def extract_keywords(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def factual_answering(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def product_description_to_ad(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def product_name_generator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.8,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def summarization(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def python_bug_fixer(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=182,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def spreadsheet_creator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def ai_language_model_tutor(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["You:"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def book_list_maker(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.52,
        presence_penalty=0.5,
        stop=["11."],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def tweet_classifier(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def airport_code_extractor(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def sql_creator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def extract_contact_info(prompt):
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


def emulate_chat(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["You:"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def mood_to_color(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[";"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def write_python_docstring(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", '"""'],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def analogy_maker(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def micro_horrow_story_creator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.8,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def third_person_converter(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def notes_to_summary(prompt):
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


def vr_fitness_idea_generator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=1,
        presence_penalty=1,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def essay_outline(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def recipe_creator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=120,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def open_ended_chat(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"],
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def sarcastic_chatbot(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


prompt = """
"Create a numbered list of turn-by-turn directions from this text:

Go south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles."
"""


def turn_by_turn_directions(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def restaurant_review_creator(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


def study_notes(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))


prompt = """
Create a list of 8 questions for my interview with a science fiction author:
"""


def interview_questions(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    print("prompt is {}".format(prompt))
    text = response["choices"][0]["text"]
    print("response text is {}".format(text))
