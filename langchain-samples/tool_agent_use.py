"""
Tool/Agent use
https://python.langchain.com/v0.1/docs/use_cases/tool_use/quickstart/
"""

import logging
import pathlib
import tempfile

import fire
from joblib import Memory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


memory = Memory(tempfile.gettempdir(), verbose=0)


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


def tool_use():
    print(f"{multiply.name=}")  # type: ignore
    print(f"{multiply.description=}")  # type: ignore
    print(f"{multiply.args=}")  # type: ignore

    out = multiply.invoke({"first_int": 4, "second_int": 5})  # type: ignore
    print(f"{out=}")

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm_with_tools = llm.bind_tools([multiply])
    msg = llm_with_tools.invoke("whats 5 times forty two")
    print(f"{msg.tool_calls=}")

    chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
    out = chain.invoke("What's four times 23")
    print(f"{out=}")


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


@memory.cache
def _agent_use(prompt) -> dict:
    tools = [multiply, add, exponentiate]
    llm = ChatOpenAI(model="gpt-4o-2024-05-13")
    agent = create_tool_calling_agent(llm, tools, prompt)  # type: ignore

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True  # type: ignore
    )
    out = agent_executor.invoke(
        {
            "input": "Take 3 to the fifth power and multiply that by the "
            "sum of twelve and three, then square the whole result"
        }
    )
    return out


def agent_use():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. {input}",
            ),
            ("human", "{agent_scratchpad}"),
        ]
    )
    out = _agent_use(prompt)
    print("agent output", "-" * 9)
    print(out["output"])


def main():
    fire.Fire(
        {
            "tool-use": tool_use,
            "agent-use": agent_use,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.INFO)
    main()
