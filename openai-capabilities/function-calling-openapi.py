"""
Function calling with an OpenAPI specification|CompletionsFunctions|Oct 15, 2023
https://cookbook.openai.com/examples/function_calling_with_an_openapi_spec
"""

import json
import logging
import pathlib
from typing import Any, Optional

import fire
import jsonref
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
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

    def add(self, content: dict):
        self.append(content)


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

    for _, methods in openapi_spec["paths"].items():
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
