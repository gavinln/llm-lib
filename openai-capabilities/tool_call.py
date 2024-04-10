"""
GPT using "tool call" or "function calling"
https://platform.openai.com/docs/guides/function-calling
"""
import json
import logging
import pathlib
import sqlite3
from typing import Any, Optional

import fire
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall
from tenacity import retry, stop_after_attempt, wait_random_exponential

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

GPT_MODEL = "gpt-3.5-turbo-0125"


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps(
            {"location": "Tokyo", "temperature": "10", "unit": unit}
        )
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps(
            {"location": "Paris", "temperature": "22", "unit": unit}
        )
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def basic_tool_call(messages, tools) -> Optional[ChatCompletion]:
    open_ai = openai.OpenAI()
    completion: ChatCompletion = open_ai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto, none - default is auto
    )
    completion_message = completion.choices[0].message
    print(completion_message)

    json_format = completion.model_dump_json(indent=2)
    print(json_format)

    available_functions = {"get_current_weather": get_current_weather}

    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls:
        messages.append(completion_message)
        for tool_call in tool_calls:
            tc: ChatCompletionMessageToolCall = tool_call
            function_name = tc.function.name
            arguments_json = tc.function.arguments
            arguments = json.loads(arguments_json)

            function_to_call = available_functions[function_name]
            unit = arguments.get("unit", None)
            if unit:
                response = function_to_call(
                    location=arguments["location"], unit=arguments["unit"]
                )
            else:
                response = function_to_call(location=arguments["location"])
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": response,
                }
            )
        second_completion: ChatCompletion = open_ai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
        )
        return second_completion
    return None


def basic_function_call():
    "basic function call"
    weather_content = (
        "What's the weather like in San Francisco, Tokyo, and Paris?"
    )
    messages = [
        {
            "role": "user",
            "content": weather_content,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "The city and state, e.g. San Francisco, CA"
                            ),
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    comp = basic_tool_call(messages, tools)
    if comp:
        comp_message = comp.choices[0].message
        print(comp_message)


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


def get_weather_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "The city and state, e.g. San Francisco, CA"
                            ),
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": (
                                "The temperature unit to use."
                                " Infer this from the users location."
                            ),
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_n_day_weather_forecast",
                "description": "Get an N-day weather forecast",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "The city and state, e.g. San Francisco, CA"
                            ),
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": (
                                "The temperature unit to use."
                                " Infer this from the users location."
                            ),
                        },
                        "num_days": {
                            "type": "integer",
                            "description": "The number of days to forecast",
                        },
                    },
                    "required": ["location", "format", "num_days"],
                },
            },
        },
    ]
    return tools


class Messages(list):
    def add_system(self, content):
        self.append({"role": "system", "content": content})

    def add_user(self, content):
        print("---user--- " + content)
        self.append({"role": "user", "content": content})

    def add(self, content):
        self.append(content)


def get_db_conn():
    db_file = SCRIPT_DIR / "data" / "chinook.db"
    return sqlite3.connect(db_file)


def ask_and_answer_place_question():
    "ask and answer place question"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user("What's the weather like today")
    chat_response: Any = chat_completion_request(messages, tools=tools)
    assistant_message = chat_response.choices[0].message
    print(assistant_message)

    messages.add(assistant_message)
    messages.add_user("I'm in Glasgow, Scotland.")
    chat_response2: Any = chat_completion_request(messages, tools=tools)
    assistant_message2 = chat_response2.choices[0].message
    print(assistant_message2)


def ask_and_answer_day_question():
    "ask and answer day question"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user(
        "What's the weather going to be like in Glasgow,"
        " Scotland over the next x days"
    )
    chat_response: Any = chat_completion_request(messages, tools=tools)
    assistant_message = chat_response.choices[0].message
    print(assistant_message)

    messages.add(assistant_message)
    messages.add_user("5 days.")
    chat_response2: Any = chat_completion_request(messages, tools=tools)
    assistant_message2 = chat_response2.choices[0].message
    print(assistant_message2)


def ask_and_force_function_call():
    "ask and force function call"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user("Give me a weather report for Toronto, Canada.")

    chat_response: Any = chat_completion_request(
        messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_n_day_weather_forecast"},
        },
    )
    assistant_message = chat_response.choices[0].message
    print(assistant_message)


def ask_and_get_function_call():
    "ask and get function call"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user("Give me a weather report for Toronto, Canada.")

    chat_response: Any = chat_completion_request(
        messages,
        tools=tools,
    )
    assistant_message = chat_response.choices[0].message
    print(assistant_message)


def ask_and_prevent_function_call():
    "ask and prevent function call"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user(
        "Give me the current weather (use Celcius) for Toronto, Canada."
    )

    chat_response: Any = chat_completion_request(
        messages, tools=tools, tool_choice="none"
    )
    assistant_message = chat_response.choices[0].message
    print(assistant_message)


def parallel_function_call():
    "parallel function call"
    tools = get_weather_tools()
    messages = Messages()
    messages.add_system(
        "Don't make assumptions about what values to plug into functions."
        " Ask for clarification if a user request is ambiguous."
    )
    messages.add_user(
        "what is the weather going to be like in San Francisco"
        " and Glasgow over the next 4 days"
    )

    chat_response: Any = chat_completion_request(messages, tools=tools)
    assistant_message = chat_response.choices[0].message
    print(assistant_message)


def get_table_names(conn):
    "Return a list of table names."
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [
        table[0]
        for table in tables.fetchall()
        if not table[0].startswith("sqlite_")
    ]


def get_column_names(conn, table_name):
    "Return a list of column names."
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    return [col[1] for col in columns]


def get_database_info(conn) -> list[dict[str, list[str]]]:
    "Return a list of dicts containing the table name and columns"
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append(
            {"table_name": table_name, "column_names": columns_names}
        )
    return table_dicts


def get_database_info_str() -> str:
    conn = get_db_conn()
    table_dicts = get_database_info(conn)
    return "\n".join(
        f"Table: {tbl['table_name']}\n"
        f"Columns: {', '.join(tbl['column_names'])}"
        for tbl in table_dicts
    )


def get_db_tools():
    database_schema_string: str = get_database_info_str()
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


def print_database_info():
    print(get_database_info_str())
    print(json.dumps(get_db_tools(), indent=2))


def ask_database(conn, query):
    "query SQLite database with a provided SQL query"
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results


def execute_function_call(message):
    function_name = message.tool_calls[0].function.name
    if function_name == "ask_database":
        query = json.loads(message.tool_calls[0].function.arguments)["query"]
        conn = get_db_conn()
        results = ask_database(conn, query)
    else:
        results = f"Error: function {function_name} does not exist"
    return results


def sql_query_function_call():
    "answer user questions by generating sql queries"
    tools = get_db_tools()
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


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "basic-function-call": basic_function_call,
            "ask-and-answer-place-question": ask_and_answer_place_question,
            "ask-and-answer-day-question": ask_and_answer_day_question,
            "ask-and-force-function-call": ask_and_force_function_call,
            "ask-and-get-function-call": ask_and_get_function_call,
            "ask-and-prevent-function-call": ask_and_prevent_function_call,
            "parallel-function-call": parallel_function_call,
            "print-database-info": print_database_info,
            "sql-query-function-call": sql_query_function_call,
        }
    )


if __name__ == "__main__":
    main()
