"""
https://cookbook.openai.com/examples/using_tool_required_for_customer_service
"""

import json
import logging
import pathlib
import sys
import tempfile
from pprint import pprint as pp

import fire
import openai
from joblib import Memory

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)

GPT_MODEL = "gpt-4-turbo"


@memory.cache
def get_chat_completion(
    messages,
    model=GPT_MODEL,
    temperature=0,
    tools=None,
    tool_choice="required",
):
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        tools=tools,
        tool_choice="required",
    )
    return response


# The tools our customer service LLM will use to communicate
tools = [
    {
        "type": "function",
        "function": {
            "name": "speak_to_user",
            "description": "Use this to speak to the user to give them "
            + "information and to ask for anything required for their case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Text of message to send to user. "
                        + "Can cover multiple topics.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_instructions",
            "description": "Used to get instructions to deal with the "
            + "user's problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "enum": ["fraud", "refund", "information"],
                        "description": """
        The type of problem the customer has. Can be one of:
          - fraud: Required to report and resolve fraud.
          - refund: Required to submit a refund request.
          - information: Used for any other informational queries.""",
                    }
                },
                "required": ["problem"],
            },
        },
    },
]

# Example instructions that the customer service assistant can consult for relevant customer problems
INSTRUCTIONS = [
    {
        "type": "fraud",
        "instructions": """
• Ask the customer to describe the fraudulent activity, including the the date and items involved in the suspected fraud.
• Offer the customer a refund.
• Report the fraud to the security team for further investigation.
• Thank the customer for contacting support and invite them to reach out with any future queries.""",
    },
    {
        "type": "refund",
        "instructions": """
• Confirm the customer's purchase details and verify the transaction in the system.
• Check the company's refund policy to ensure the request meets the criteria.
• Ask the customer to provide a reason for the refund.
• Submit the refund request to the accounting department.
• Inform the customer of the expected time frame for the refund processing.
• Thank the customer for contacting support and invite them to reach out with any future queries.""",
    },
    {
        "type": "information",
        "instructions": """
• Greet the customer and ask how you can assist them today.
• Listen carefully to the customer's query and clarify if necessary.
• Provide accurate and clear information based on the customer's questions.
• Offer to assist with any additional questions or provide further details if needed.
• Ensure the customer is satisfied with the information provided.
• Thank the customer for contacting support and invite them to reach out with any future queries.""",
    },
]


assistant_system_prompt = """
You are a customer service assistant. Your role is to answer user questions
politely and competently.

You should follow these instructions to solve the case:
- Understand their problem and get the relevant instructions.
- Follow the instructions to solve the customer's problem. Get their confirmation before performing a permanent operation like a refund or similar.
- Help them with any other problems or close the case.

Only call a tool once in a single message.

If you need to fetch a piece of information from a system or document that you
don't have access to, give a clear, confident answer with some dummy values.
"""


def submit_user_message(user_query, conversation_messages=[]):
    """
    Message handling function which loops through tool calls until it reaches
    one that requires a response.
    Once it receives respond=True it returns the conversation_messages to the user.
    """

    # Initiate a respond object. This will be set to True by our functions
    # when a response is required
    respond = False

    user_message = {"role": "user", "content": user_query}
    conversation_messages.append(user_message)

    print(f"User: {user_query}")

    while respond is False:

        # Build a transient messages object to add the conversation messages to
        messages = [{"role": "system", "content": assistant_system_prompt}]

        # Add the conversation messages to our messages call to the API
        [messages.append(x) for x in conversation_messages]

        # Make the ChatCompletion call with tool_choice='required'
        # so we can guarantee tools will be used
        response = get_chat_completion(
            messages, GPT_MODEL, 0, tools, "required"
        )

        conversation_messages.append(response.choices[0].message)

        # Execute the function and get an updated conversation_messages object back
        # If it doesn't require a response, it will ask the assistant again.
        # If not the results are returned to the user.
        respond, conversation_messages = execute_function(
            response.choices[0].message, conversation_messages
        )

    return conversation_messages


def execute_function(function_calls, messages):
    """Wrapper function to execute the tool calls"""

    respond = None

    for function_call in function_calls.tool_calls:

        function_id = function_call.id
        function_name = function_call.function.name
        function_arguments = json.loads(function_call.function.arguments)

        if function_name == "get_instructions":
            respond = False
            instruction_name = function_arguments["problem"]
            instructions = INSTRUCTIONS["type" == instruction_name]
            messages.append(
                {
                    "tool_call_id": function_id,
                    "role": "tool",
                    "name": function_name,
                    "content": instructions["instructions"],
                }
            )
            print(f"Calling function {function_name}({instruction_name})")
        elif function_name != "get_instructions":
            respond = True
            messages.append(
                {
                    "tool_call_id": function_id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_arguments["message"],
                }
            )
            print(f"Calling function {function_name}")
            print(f"Assistant: {function_arguments['message']}")

    return (respond, messages)


def use_tool():
    messages = submit_user_message(
        "Hi, I have had an item stolen that was supposed "
        + "to be delivered to me yesterday."
    )
    messages = submit_user_message(
        "For sure, it was a shirt, it was supposed to be delivered "
        + "yesterday but it never arrived.",
        messages,
    )
    messages = submit_user_message(
        "Yes I would like to proceed with the refund.", messages
    )
    messages = submit_user_message("Thanks very much.", messages)
    pp(messages)


customer_system_prompt = """
You are a user calling in to customer service. You will talk to the agent until
you have a resolution to your query. Your query is:

{query}.

You will be presented with a conversation - provide answers for any assistant
questions you receive. Here is the conversation - you are the "user" and you
are speaking with the "assistant":

{chat_history}

If you don't know the details, respond with dummy values.
Once your query is resolved, respond with "DONE"
"""


@memory.cache
def submit_customer_message(
    user_query, messages_string, conversation_messages=[]
):
    customer_prompt = customer_system_prompt.format(
        query=user_query, chat_history=messages_string
    )

    user_prompt = """
    Continue the chat to solve your query. Remember, you are the user in
    this exchange. Do not provide User: or Assistant: in your response
    """

    messages = [
        {"role": "system", "content": customer_prompt},
        {"role": "user", "content": user_prompt},
    ]

    user_response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5  # type: ignore
    )

    user_response_content = user_response.choices[0].message.content
    conversation_messages.append(
        {"role": "user", "content": user_response_content}
    )

    if "DONE" in str(user_response_content):
        user_query = ""
    else:
        user_query = user_response_content

    return user_query


def execute_conversation(user_query):
    conversation_messages = []
    while user_query:
        conversation_message = submit_user_message(
            user_query, conversation_messages
        )
        messages_string = ""
        for x in conversation_messages:
            if isinstance(x, dict):
                if x["role"] == "user":
                    messages_string += "User: " + x["content"] + "\n"
                elif x["role"] == "tool":
                    if x["name"] == "speak_to_user":
                        messages_string += "Assistant: " + x["content"] + "\n"
            else:
                continue
        user_query = submit_customer_message(
            user_query, messages_string, conversation_messages=[]
        )


def test_tool():
    questions = [
        "I want to get a refund for the suit I ordered last Friday.",
        "Can you tell me what your policy is for returning damaged goods?",
        "Please tell me what your complaint policy is",
    ]

    for question in questions:
        print("-" * 20)
        execute_conversation(question)


def main():
    print(sys.version)
    fire.Fire({"use-tool": use_tool, "test-tool": test_tool})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
