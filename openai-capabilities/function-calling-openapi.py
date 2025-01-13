"""
function calling using openapi specifications
https://cookbook.openai.com/examples/function_calling_with_an_openapi_spec
"""

import json
import logging
import pathlib
import pprint as pp
import sqlite3
from typing import Any, Optional

import fire
import jsonref
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

# GPT_MODEL = "gpt-3.5-turbo-0125"
GPT_MODEL = "gpt-4-turbo"


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
)
def chat_completion_request(
    messages, tools: Any = None, tool_choice: Any = None, model=GPT_MODEL
) -> Optional[ChatCompletion]:
    try:
        response: ChatCompletion = openai.OpenAI().chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
    return None


class Messages(list):
    def add_system(self, content: str):
        print("---user--- " + content)
        self.append({"role": "system", "content": content})

    def add_user(self, content: str):
        print("---user--- " + content)
        self.append({"role": "user", "content": content})

    def add_tool(self, content: str):
        print("---tool--- " + content)
        self.append({"role": "user", "content": content})

    def add(self, content: dict):
        self.append(content)


def get_db_tools_example():
    "does not work as there is no database schema"
    database_schema_string: str = ""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": (
                    "Use this function to answer user questions about music."
                    " Input should be a fully formed SQL query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                    SQL query extracting info to answer the user's question.
                    SQL should be written using this database schema:
                    {database_schema_string}
                    The query should be returned in plain text, not in JSON.
                            """,
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools


def ask_database(conn, query):
    "query SQLite database with a provided SQL query"
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results


def execute_function_call(message):
    function_name = message.tool_calls[0].function.name
    return ""


def sql_query_function_call():
    "answer user questions by generating sql queries"
    tools = get_db_tools_example()
    messages = Messages()
    messages.add_system(
        "Answer user questions by generating SQL queries"
        " against the Chinook Music Database."
    )
    messages.add_user("Hi, who are the top 5 artists by number of tracks?")
    chat_response = chat_completion_request(messages, tools)
    if not chat_response:
        return

    assistant_message: ChatCompletionMessage = chat_response.choices[0].message
    print(assistant_message)

    if not assistant_message.tool_calls:
        return

    assistant_message.content = str(assistant_message.tool_calls[0].function)
    messages.add(
        {
            "role": assistant_message.role,
            "content": assistant_message.content,
        }
    )
    result = execute_function_call(assistant_message)
    print(result)

    messages.clear()
    messages.add_system(
        "Answer user questions by generating SQL queries"
        " against the Chinook Music Database."
    )
    messages.add_user("What is the name of the album with the most tracks?")

    chat_response2 = chat_completion_request(messages, tools)
    if not chat_response2:
        return

    assistant_message2: ChatCompletionMessage = chat_response2.choices[
        0
    ].message
    print(assistant_message2)

    if not assistant_message2.tool_calls:
        return

    assistant_message2.content = str(assistant_message2.tool_calls[0].function)
    messages.add(
        {
            "role": assistant_message2.role,
            "content": assistant_message2.content,
        }
    )
    result2 = execute_function_call(assistant_message2)
    print(result2)


def get_openai_spec_file():
    return SCRIPT_DIR / "example_events_openapi.json"


def get_openai_spec():
    spec_file = get_openai_spec_file()
    spec = ""
    with spec_file.open() as f:
        spec = jsonref.loads(f.read())
    return spec


def openapi_to_functions(openapi_spec):
    functions = []

    for path, methods in openapi_spec["paths"].items():
        for method, spec_with_ref in methods.items():
            # 1. Resolve JSON references.
            spec: dict = jsonref.replace_refs(spec_with_ref)  # type: ignore

            # 2. Extract a name for the functions.
            function_name = spec.get("operationId")

            # 3. Extract a description and parameters.
            desc = spec.get("description") or spec.get("summary", "")

            schema = {"type": "object", "properties": {}}

            req_body = (
                spec.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if req_body:
                schema["properties"]["requestBody"] = req_body

            params = spec.get("parameters", [])
            if params:
                param_properties = {
                    param["name"]: param["schema"]
                    for param in params
                    if "schema" in param
                }
                schema["properties"]["parameters"] = {
                    "type": "object",
                    "properties": param_properties,
                }

            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )

    return functions


def function_calling_openapi():
    spec = get_openai_spec()
    functions = openapi_to_functions(spec)
    for function in functions:
        print(json.dumps(function, indent=2))

    SYSTEM_MESSAGE = """
    You are a helpful assistant.
    Respond to the following prompt by using function_call and then
    summarize actions.
    Ask for clarification if a user request is ambiguous.
    """

    USER_MESSAGE = """
    Instruction: Get all the events.
    Then create a new event named AGI Party.
    Then delete event with id 2456.
    """

    MAX_CALLS = 3

    messages = Messages()
    messages.add_system(SYSTEM_MESSAGE)
    messages.add_user(USER_MESSAGE)

    tools = functions
    tool_choice = "auto"

    num_calls = 0
    chat_response = chat_completion_request(messages, tools, tool_choice)
    while num_calls < MAX_CALLS:
        if not chat_response:
            return
        assistant_message: ChatCompletionMessage = chat_response.choices[
            0
        ].message
        num_tool_calls = 0
        if assistant_message.tool_calls:
            num_tool_calls = len(assistant_message.tool_calls)
            print("There are {} tool calls".format(num_tool_calls))
            for tool_call in assistant_message.tool_calls:
                print(tool_call)
            if num_tool_calls < MAX_CALLS:
                messages.add_user(
                    f"Only {num_tool_calls} tool call generated. Please generate all"
                )
                chat_response = chat_completion_request(
                    messages, tools, tool_choice
                )
            else:
                num_calls += num_tool_calls


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "function-calling-openapi": function_calling_openapi,
            "sql-query-function-call": sql_query_function_call,
        }
    )


if __name__ == "__main__":
    main()
