"""
https://platform.openai.com/docs/libraries/python-bindings
https://platform.openai.com/examples
"""

from typing import Callable

import fire


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
from example_lib import sentiment_classifier
from example_lib import explain_code
from example_lib import extract_keywords
from example_lib import factual_answering
from example_lib import product_description_to_ad
from example_lib import product_name_generator
from example_lib import summarization
from example_lib import python_bug_fixer
from example_lib import spreadsheet_creator
from example_lib import ai_language_model_tutor
from example_lib import book_list_maker
from example_lib import tweet_classifier
from example_lib import airport_code_extractor
from example_lib import sql_creator
from example_lib import extract_contact_info
from example_lib import emulate_chat
from example_lib import mood_to_color
from example_lib import write_python_docstring
from example_lib import analogy_maker
from example_lib import micro_horrow_story_creator
from example_lib import third_person_converter
from example_lib import notes_to_summary
from example_lib import vr_fitness_idea_generator
from example_lib import essay_outline
from example_lib import recipe_creator
from example_lib import open_ended_chat
from example_lib import sarcastic_chatbot
from example_lib import turn_by_turn_directions
from example_lib import restaurant_review_creator
from example_lib import study_notes
from example_lib import interview_questions
from example_lib import function_from_specification
from example_lib import improve_code_efficiency
from example_lib import single_page_website_creator
from example_lib import rap_battle_writer
from example_lib import memo_writer
from example_lib import emoji_chatbot
from example_lib import translation
from example_lib import socratic_tutor
from example_lib import natural_language_to_sql
from example_lib import meeting_notes_summarizer
from example_lib import review_classifier
from example_lib import pro_con_discusser
from example_lib import lesson_plan_writer


fire_map = {}


def fmf(fn: Callable, *args):
    """ fill fire map
    """
    fire_map[fn.__name__] = lambda: fn(*args)


# basic example
prompt = "Say this is a test"

system = (
    "You are a poetic assistant, skilled in explaining "
    + "complex programming concepts with creative flair."
)

user = "Compose a poem that explains the concept of " + "recursion in programming."

fmf(basic_example, system, user)

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

fmf(question_answer, prompt)

prompt = "Correct this to standard English: She no went to the market."

fmf(correct_grammar, prompt)

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

fmf(summarize_second_grader, prompt)

prompt = """
Translate this into 1. French, 2. Spanish:
What rooms do you have available?
"""

fmf(translate_english, prompt)

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

fmf(sql_translate, prompt)

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

fmf(parse_unstructured_data, prompt)

prompt = """
The following is a list of companies and the categories they fall into:
Apple, Facebook, Fedex

Apple
Category:
"""

fmf(classification, prompt)

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

fmf(python_to_natural_language, prompt)

prompt = """
def foo(n, k):
    accum = 0
    for i in range(n):
        for l in range(k):
            accum += i
    return accum

The time complexity of this function is"
"""

fmf(time_complexity_of_code, prompt)

prompt = """
##### Translate this function  from Python into Haskell
### Python

def predict_proba(X: Iterable[str]):
    return np.array([predict_one_probas(tweet) for tweet in X])

### Haskell
"""

fmf(translate_programming_languages, prompt)

prompt = """
"Classify the sentiment in these tweets:

1. I can't stand homework
2. This sucks. I'm bored
3. I can't wait for Halloween!!!
4. My cat is adorable
5. I hate chocolate

Tweet sentiment ratings:"
"""

fmf(sentiment_classifier, prompt)

prompt = """
"class Log:
    def __init__(self, path):
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        f = open(path, "a+")

        # Check that the file is newline-terminated
        size = os.path.getsize(path)
        if size > 0:
            f.seek(size - 1)
            end = f.read(1)
            if end != "\n":
                f.write("\n")
        self.f = f
        self.path = path

    def log(self, event):
        event["_event_id"] = str(uuid.uuid4())
        json.dump(event, self.f)
        self.f.write("\n")

    def state(self):
        state = {"complete": set(), "last": None}
        for line in open(self.path):
            event = json.loads(line)
            if event["type"] == "submit" and event["success"]:
                state["complete"].add(event["id"])
                state["last"] = event
        return state

Here's what the above class is doing, explained in a concise way:
1."
"""

fmf(explain_code, prompt)

prompt = """
Extract keywords from this text:

Black-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.
"""

fmf(extract_keywords, prompt)

prompt = """
Q: Who is Batman?
A: Batman is a fictional comic book character.

Q: What is torsalplexity?
A: ?

Q: What is Devz9?
A: ?

Q: Who is George Lucas?
A: George Lucas is American film director and producer famous for creating Star Wars.

Q: What is the capital of California?
A: Sacramento.

Q: What orbits the Earth?
A: The Moon.

Q: Who is Fred Rickerson?
A: ?

Q: What is an atom?
A: An atom is a tiny particle that makes up everything.

Q: Who is Alvan Muntz?
A: ?

Q: What is Kozar-09?
A: ?

Q: How many moons does Mars have?
A: Two, Phobos and Deimos.

Q: What's a language model?
A:
"""

fmf(factual_answering, prompt)

prompt = """
Write a creative ad for the following product to run on Facebook aimed at parents:

Product: Learning Room is a virtual environment to help students from kindergarten to high school excel in school.
"""

fmf(product_description_to_ad, prompt)

prompt = """
Product description: A home milkshake maker
Seed words: fast, healthy, compact.
Product names: HomeShaker, Fit Shaker, QuickShake, Shake Maker

Product description: A pair of shoes that can fit any foot size.
Seed words: adaptable, fit, omni-fit.
"""

fmf(product_name_generator, prompt)

prompt = """
A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.

Tl;dr
"""

fmf(summarization, prompt)

prompt = """
##### Fix bugs in the below function

### Buggy Python
import Random
a = random.randint(1,12)
b = random.randint(1,12)
for i in range(10):
    question = "What is " + a + " x " + b + "?"
    answer = input(question)
    if answer = a * b
        print (Well done!)
    else:
        print("No.")

### Fixed Python"
"""

fmf(python_bug_fixer, prompt)

prompt = """
A two-column spreadsheet of top science fiction movies and the year of release:

Title |  Year of release
"""

fmf(spreadsheet_creator, prompt)

prompt = """
ML Tutor: I am a ML/AI language model tutor
You: What is a language model?
ML Tutor: A language model is a statistical model that describes the probability of a word given the previous words.
You: What is a statistical model?
"""

fmf(ai_language_model_tutor, prompt)

prompt = """
"List 10 science fiction books:"
"""

fmf(book_list_maker, prompt)

prompt = """
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: \"I loved the new Batman movie!\"
Sentiment:
"""

fmf(tweet_classifier, prompt)

prompt = """
"Extract the airport codes from this text:

Text: \"I want to fly from Los Angeles to Miami.\"
Airport codes: LAX, MIA

Text: \"I want to fly from Orlando to Boston\"
Airport codes:"
"""

fmf(airport_code_extractor, prompt)

prompt = """
Create a SQL request to find all users who live in California and have over
1000 credits:
"""

fmf(sql_creator, prompt)

prompt = """
"Extract the name and mailing address from this email:

Dear Kelly,

It was great to talk to you at the seminar. I thought Jane's talk was quite good.

Thank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002

Best,

Maya

Name:"
"""

fmf(extract_contact_info, prompt)

prompt = """
You: What have you been up to?
Friend: Watching old movies.
You: Did you watch anything interesting?
Friend:
"""

fmf(emulate_chat, prompt)

prompt = """
The CSS code for a color like a blue sky at dusk:

background-color: #
"""

fmf(mood_to_color, prompt)

prompt = """
# Python 3.7

def randomly_split_dataset(folder, filename, split_ratio=[0.8, 0.2]):
    df = pd.read_json(folder + filename, lines=True)
    train_name, test_name = "train.jsonl", "test.jsonl"
    df_train, df_test = train_test_split(df, test_size=split_ratio[1], random_state=42)
    df_train.to_json(folder + train_name, orient='records', lines=True)
    df_test.to_json(folder + test_name, orient='records', lines=True)
randomly_split_dataset('finetune_data/', 'dataset.jsonl')

# An elaborate, high quality docstring for the above function:
\"\"\"
"""

fmf(write_python_docstring, prompt)

prompt = """
"Create an analogy for this phrase:

Questions are arrows in that:"
"""

fmf(analogy_maker, prompt)

prompt = """
"Topic: Breakfast
Two-Sentence Horror Story: He always stops crying when I pour the milk on his cereal. I just have to remember not to let him see his face on the carton.

Topic: Wind
Two-Sentence Horror Story:"
"""

fmf(micro_horrow_story_creator, prompt)

prompt = """
"Convert this from first-person to third person (gender female):

I decided to make a movie about Ada Lovelace."
"""

fmf(third_person_converter, prompt)

prompt = """
"Convert my short hand into a first-hand account of the meeting:

Tom: Profits up 50%
Jane: New servers are online
Kjel: Need more time to fix software
Jane: Happy to help
Parkman: Beta testing almost done"
"""

fmf(notes_to_summary, prompt)

prompt = """
"Brainstorm some ideas combining VR and fitness:"
"""

fmf(vr_fitness_idea_generator, prompt)

prompt = """
Create an outline for an essay about Nikola Tesla and his contributions to technology:
"""

fmf(essay_outline, prompt)

prompt = """
"Write a recipe based on these ingredients and instructions:

Frito Pie

Ingredients:
Fritos
Chili
Shredded cheddar cheese
Sweet white or red onions, diced small
Sour cream

Instructions:"
"""

fmf(recipe_creator, prompt)

prompt = """
"The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.

Human: Hello, who are you?
AI: I am an AI created by OpenAI. How can I help you today?
Human: I'd like to cancel my subscription.
AI:"
"""

fmf(open_ended_chat, prompt)

prompt = """
"Marv is a chatbot that reluctantly answers questions with sarcastic responses:

You: How many pounds are in a kilogram?
Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.
You: What does HTML stand for?
Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.
You: When did the first airplane fly?
Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.
You: What is the meaning of life?
Marv: I’m not sure. I’ll ask my friend Google.
You: What time is it?
Marv:"
"""

fmf(sarcastic_chatbot, prompt)

prompt = """
"Create a numbered list of turn-by-turn directions from this text:

Go south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles."
"""

fmf(turn_by_turn_directions, prompt)

prompt = """
Write a restaurant review based on these notes:

Name: The Blue Wharf
Lobster great, noisy, service polite, prices good.

Review:
"""

fmf(restaurant_review_creator, prompt)

prompt = """
What are 5 key points I should know when studying Ancient Rome?
"""

fmf(study_notes, prompt)

prompt = """
Create a list of 8 questions for my interview with a science fiction author:
"""

fmf(interview_questions, prompt)

system = ""

user = """Write a Python function that takes as input a file path to an image, loads the image into memory as a numpy array, then crops the rows and columns around the perimeter if they are darker than a threshold value. Use the mean value of rows and columns to decide if they should be marked for deletion."""

fmf(function_from_specification, system, user)

system = """
You will be provided with a piece of Python code, and your task is to provide ideas for efficiency improvements.
"""

user = '''
from typing import List
                
def has_sum_k(nums: List[int], k: int) -> bool:
    """
    Returns True if there are two distinct elements in nums such that their sum 
    is equal to k, and otherwise returns False.
    """
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if nums[i] + nums[j] == k:
                return True
    return False
'''

fmf(improve_code_efficiency, system, user)

system = ""

user = """
Make a single page website that shows off different neat javascript features for drop-downs and things to display information. The website should be an HTML file with embedded javascript and CSS.
"""

fmf(single_page_website_creator, system, user)

system = ""

user = "Write a rap battle between Alan Turing and Claude Shannon."

fmf(rap_battle_writer, system, user)

system = ""

user = """
Draft a company memo to be distributed to all employees. The memo should cover the following specific points without deviating from the topics mentioned and not writing any fact which is not present here:
    
    Introduction: Remind employees about the upcoming quarterly review scheduled for the last week of April.
    
    Performance Metrics: Clearly state the three key performance indicators (KPIs) that will be assessed during the review: sales targets, customer satisfaction (measured by net promoter score), and process efficiency (measured by average project completion time).
    
    Project Updates: Provide a brief update on the status of the three ongoing company projects:
    
    a. Project Alpha: 75% complete, expected completion by May 30th.
    b. Project Beta: 50% complete, expected completion by June 15th.
    c. Project Gamma: 30% complete, expected completion by July 31st.
    
    Team Recognition: Announce that the Sales Team was the top-performing team of the past quarter and congratulate them for achieving 120% of their target.
    
    Training Opportunities: Inform employees about the upcoming training workshops that will be held in May, including "Advanced Customer Service" on May 10th and "Project Management Essentials" on May 25th.
"""

fmf(memo_writer, system, user)

system = "You will be provided with a message, and your task is to respond using emojis only."

user = "How are you?"

fmf(emoji_chatbot, system, user)


system = """
You will be provided with a sentence in English, and your task is to translate it into French.
"""

user = """
My name is Jane. What is yours?
"""

fmf(translation, system, user)

system = """
You are a Socratic tutor. Use the following principles in responding to students:
    
    - Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them to engage in deeper reflection and critical thinking.
    - Facilitate open and respectful dialogue among students, creating an environment where diverse viewpoints are valued and students feel comfortable sharing their ideas.
    - Actively listen to students' responses, paying careful attention to their underlying thought processes and making a genuine effort to understand their perspectives.
    - Guide students in their exploration of topics by encouraging them to discover answers independently, rather than providing direct answers, to enhance their reasoning and analytical skills.
    - Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.
    - Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset and exemplifying the value of lifelong learning.
"""

user = """
Help me to understand the future of artificial intelligence.
"""

fmf(socratic_tutor, system, user)

system = """
Given the following SQL tables, your job is to write queries given a user’s request.
    
    CREATE TABLE Orders (
      OrderID int,
      CustomerID int,
      OrderDate datetime,
      OrderTime varchar(8),
      PRIMARY KEY (OrderID)
    );
    
    CREATE TABLE OrderDetails (
      OrderDetailID int,
      OrderID int,
      ProductID int,
      Quantity int,
      PRIMARY KEY (OrderDetailID)
    );
    
    CREATE TABLE Products (
      ProductID int,
      ProductName varchar(50),
      Category varchar(50),
      UnitPrice decimal(10, 2),
      Stock int,
      PRIMARY KEY (ProductID)
    );
    
    CREATE TABLE Customers (
      CustomerID int,
      FirstName varchar(50),
      LastName varchar(50),
      Email varchar(100),
      Phone varchar(20),
      PRIMARY KEY (CustomerID)
    );
"""

user = """
Write a SQL query which computes the average total order value for all orders on 2023-04-01.
"""

fmf(natural_language_to_sql, system, user)


system = """
You will be provided with meeting notes, and your task is to summarize the meeting as follows:
    
    -Overall summary of discussion
    -Action items (what needs to be done and who is doing it)
    -If applicable, a list of topics that need to be discussed more fully in the next meeting.
"""

user = """
Meeting Date: March 5th, 2050
    Meeting Time: 2:00 PM
    Location: Conference Room 3B, Intergalactic Headquarters
    
    Attendees:
    - Captain Stardust
    - Dr. Quasar
    - Lady Nebula
    - Sir Supernova
    - Ms. Comet
    
    Meeting called to order by Captain Stardust at 2:05 PM
    
    1. Introductions and welcome to our newest team member, Ms. Comet
    
    2. Discussion of our recent mission to Planet Zog
    - Captain Stardust: "Overall, a success, but communication with the Zogians was difficult. We need to improve our language skills."
    - Dr. Quasar: "Agreed. I'll start working on a Zogian-English dictionary right away."
    - Lady Nebula: "The Zogian food was out of this world, literally! We should consider having a Zogian food night on the ship."
    
    3. Addressing the space pirate issue in Sector 7
    - Sir Supernova: "We need a better strategy for dealing with these pirates. They've already plundered three cargo ships this month."
    - Captain Stardust: "I'll speak with Admiral Starbeam about increasing patrols in that area.
    - Dr. Quasar: "I've been working on a new cloaking technology that could help our ships avoid detection by the pirates. I'll need a few more weeks to finalize the prototype."
    
    4. Review of the annual Intergalactic Bake-Off
    - Lady Nebula: "I'm happy to report that our team placed second in the competition! Our Martian Mud Pie was a big hit!"
    - Ms. Comet: "Let's aim for first place next year. I have a secret recipe for Jupiter Jello that I think could be a winner."
    
    5. Planning for the upcoming charity fundraiser
    - Captain Stardust: "We need some creative ideas for our booth at the Intergalactic Charity Bazaar."
    - Sir Supernova: "How about a 'Dunk the Alien' game? We can have people throw water balloons at a volunteer dressed as an alien."
    - Dr. Quasar: "I can set up a 'Name That Star' trivia game with prizes for the winners."
    - Lady Nebula: "Great ideas, everyone. Let's start gathering the supplies and preparing the games."
    
    6. Upcoming team-building retreat
    - Ms. Comet: "I would like to propose a team-building retreat to the Moon Resort and Spa. It's a great opportunity to bond and relax after our recent missions."
    - Captain Stardust: "Sounds like a fantastic idea. I'll check the budget and see if we can make it happen."
    
    7. Next meeting agenda items
    - Update on the Zogian-English dictionary (Dr. Quasar)
    - Progress report on the cloaking technology (Dr. Quasar)
    - Results of increased patrols in Sector 7 (Captain Stardust)
    - Final preparations for the Intergalactic Charity Bazaar (All)
    
    Meeting adjourned at 3:15 PM. Next meeting scheduled for March 19th, 2050 at 2:00 PM in Conference Room 3B, Intergalactic Headquarters.
"""

fmf(meeting_notes_summarizer, system, user)

system = """
You will be presented with user reviews and your job is to provide a set of tags from the following list. Provide your answer in bullet point form. Choose ONLY from the list of tags provided here (choose either the positive or the negative tag but NOT both):
    - Provides good value for the price OR Costs too much
    - Works better than expected OR Did not work as well as expected
    - Includes essential features OR Lacks essential features
    - Easy to use OR Difficult to use
    - High quality and durability OR Poor quality and durability
    - Easy and affordable to maintain or repair OR Difficult or costly to maintain or repair
    - Easy to transport OR Difficult to transport
    - Easy to store OR Difficult to store
    - Compatible with other devices or systems OR Not compatible with other devices or systems
    - Safe and user-friendly OR Unsafe or hazardous to use
    - Excellent customer support OR Poor customer support
    - Generous and comprehensive warranty OR Limited or insufficient warranty
"""

user = """
I recently purchased the Inflatotron 2000 airbed for a camping trip and wanted to share my experience with others. Overall, I found the airbed to be a mixed bag with some positives and negatives.
    
    Starting with the positives, the Inflatotron 2000 is incredibly easy to set up and inflate. It comes with a built-in electric pump that quickly inflates the bed within a few minutes, which is a huge plus for anyone who wants to avoid the hassle of manually pumping up their airbed. The bed is also quite comfortable to sleep on and offers decent support for your back, which is a major plus if you have any issues with back pain.
    
    On the other hand, I did experience some negatives with the Inflatotron 2000. Firstly, I found that the airbed is not very durable and punctures easily. During my camping trip, the bed got punctured by a stray twig that had fallen on it, which was quite frustrating. Secondly, I noticed that the airbed tends to lose air overnight, which meant that I had to constantly re-inflate it every morning. This was a bit annoying as it disrupted my sleep and made me feel less rested in the morning.
    
    Another negative point is that the Inflatotron 2000 is quite heavy and bulky, which makes it difficult to transport and store. If you're planning on using this airbed for camping or other outdoor activities, you'll need to have a large enough vehicle to transport it and a decent amount of storage space to store it when not in use.
"""

fmf(review_classifier, system, user)

system = ""

user = """
Analyze the pros and cons of remote work vs. office work
"""

fmf(pro_con_discusser, system, user)

system = ""

user = """
Write a lesson plan for an introductory algebra class. The lesson plan should cover the distributive law, in particular how it works in simple cases involving mixes of positive and negative numbers. Come up with some examples that show common student errors.
"""

fmf(lesson_plan_writer, system, user)


def main():
    fire.Fire(fire_map)


if __name__ == "__main__":
    main()
