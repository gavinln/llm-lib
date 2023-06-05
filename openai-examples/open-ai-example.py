"""
https://platform.openai.com/docs/libraries/python-bindings
"""

from example_lib import basic_example
from example_lib import question_answer
from example_lib import correct_grammar 
from example_lib import summarize_second_grader
from example_lib import translate_english
from example_lib import sql_translate
from example_lib import parse_unstructured_data
from example_lib import classification
from example_lib import python_to_natural_language
from example_lib import time_complexity_of_code
from example_lib import translate_programming_languages

# basic example
prompt = "Say this is a test"

# basic_example(prompt)

# question_answer
prompt = """

I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q: Where is the Valley of Kings?
A:

"""

# question_answer(prompt)

prompt = "Correct this to standard English: She no went to the market."

# correct_grammar(prompt)

prompt = """
Summarize this for a second-grade student:

Jupiter is the fifth planet from the Sun and the largest in the Solar System.
It is a gas giant with a mass one-thousandth that of the Sun, but
two-and-a-half times that of all the other planets in the Solar System
combined. Jupiter is one of the brightest objects visible to the naked eye 
in the night sky, and has been known to ancient civilizations since before
recorded history. It is named after the Roman god Jupiter.[19] When viewed
from Earth, Jupiter can be bright enough for its reflected light to cast
visible shadows,[20] and is on average the third-brightest natural object
in the night sky after the Moon and Venus."

"""

# summarize_second_grader(prompt)


prompt="""
Translate this into 1. French, 2. Spanish:
What rooms do you have available?
"""

# translate_english(prompt)

prompt = """
# Postgres SQL tables, with their properties:
#
# Employee(id, name, department_id)
# Department(id, name, address)
# Salary_Payments(id, employee_id, amount, date)
#
# A query to list the names of the departments which employed more than 10 employees in the last 3 months
SELECT
"""

# sql_translate(prompt)

prompt = """
A table summarizing the fruits from Goocrux:

There are many fruits that were found on the recently discovered planet
Goocrux. There are neoskizzles that grow there, which are purple and taste
like candy. There are also loheckles, which are a grayish blue fruit and
are very tart, a little bit like a lemon. Pounits are a bright green color
and are more savory than sweet. There are also plenty of loopnovas which
are a neon pink flavor and taste like cotton candy. Finally, there are
fruits called glowls, which have a very sour and bitter taste which is
acidic and caustic, and a pale orange tinge to them.

| Fruit | Color | Flavor |
"""

# parse_unstructured_data(prompt)


prompt = """
The following is a list of companies and the categories they fall into:
Apple, Facebook, Fedex

Apple
Category:
"""

# classification(prompt)

prompt = """
# Python 3 
def remove_common_prefix(x, prefix, ws_prefix): 
    x["completion"] = x["completion"].str[len(prefix) :] 
    if ws_prefix: 
        # keep the single whitespace as prefix 
        x["completion"] = " " + x["completion"] 
    return x 

# Explanation of what the code does
#
"""

# python_to_natural_language(prompt)


prompt = """
def foo(n, k):
    accum = 0
    for i in range(n):
        for l in range(k):
            accum += i
    return accum

The time complexity of this function is"
"""

# time_complexity_of_code(prompt)


prompt = """
##### Translate this function  from Python into Haskell
### Python
    
def predict_proba(X: Iterable[str]):
    return np.array([predict_one_probas(tweet) for tweet in X])
    
### Haskell
"""

translate_programming_languages(prompt)
