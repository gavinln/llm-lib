"""
https://platform.openai.com/docs/libraries/python-bindings
"""

import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "Say this is a test"

response = openai.Completion.create(
    model="text-davinci-003", prompt=prompt,
    temperature=0, max_tokens=7
)
print('prompt is {}'.format(prompt))
print('response text is {}'.format(response['choices'][0]['text']))
