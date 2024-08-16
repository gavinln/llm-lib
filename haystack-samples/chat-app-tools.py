"""
https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling
"""

import json
import logging
import pathlib
import pprint as pp
import sys
import tempfile

import fire
import gradio as gr
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from joblib import Memory

memory = Memory(tempfile.gettempdir(), verbose=0)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))

RAG_PIPE = None


def example_create_chat_generator():
    system_prompt = (
        "Always respond in German even if some input "
        + " data is in other languages."
    )
    user_prompt = "What's Natural Language Processing? Be brief."
    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(user_prompt),
    ]

    print(f"{system_prompt=}")
    print(f"{user_prompt=}")
    chat_generator = OpenAIChatGenerator(model="gpt-3.5-turbo")
    out = chat_generator.run(messages=messages)
    pp.pprint(out)


@memory.cache
def create_indexed_documents():
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
        Document(content="My name is Marta and I live in Madrid."),
        Document(content="My name is Harry and I live in London."),
    ]

    document_store = InMemoryDocumentStore()
    embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    writer = DocumentWriter(document_store=document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=embedder, name="doc_embedder")
    indexing_pipeline.add_component(instance=writer, name="doc_writer")

    indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

    _ = indexing_pipeline.run({"doc_embedder": {"documents": documents}})
    return document_store


def create_rag_pipeline(document_store: InMemoryDocumentStore):
    template = """
    Answer the questions based on the given context.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{ question }}
    Answer:
    """
    rag_pipe = Pipeline()
    embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    rag_pipe.add_component("embedder", embedder)
    rag_pipe.add_component("retriever", retriever)
    rag_pipe.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipe.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))

    rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
    rag_pipe.connect("retriever", "prompt_builder.documents")
    rag_pipe.connect("prompt_builder", "llm")

    query = "Where does Mark live?"
    print(f"{query=}")
    out = rag_pipe.run(
        {"embedder": {"text": query}, "prompt_builder": {"question": query}}
    )
    pp.pprint(out)
    return rag_pipe


WEATHER_INFO = {
    "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
    "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
    "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
    "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
}


def get_current_weather(location: str):
    if location in WEATHER_INFO:
        return WEATHER_INFO[location]
    else:
        return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}


def get_rag_pipeline_func_spec():
    return (
        {
            "type": "function",
            "function": {
                "name": "rag_pipeline_func",
                "description": "Get information about where people live",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to use in the search. "
                            + "Infer this from the user's message. It "
                            + "should be a question or a statement",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    )


def get_current_weather_spec():
    return (
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
                            "description": "The city and state, e.g. San "
                            + "Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    )


def rag_pipeline_func(query: str):
    assert RAG_PIPE, "RAG_PIPE is not initialized"
    result = RAG_PIPE.run(
        {
            "embedder": {"text": query},
            "prompt_builder": {"question": query},
        }
    )
    return {"reply": result["llm"]["replies"][0]}


def get_function_response(response) -> tuple[str, str]:
    assert "replies" in response
    assert len(response["replies"]) > 0

    function_call = json.loads(response["replies"][0].content)[0]
    function_name = function_call["function"]["name"]
    function_args = json.loads(function_call["function"]["arguments"])
    print("Function Name:", function_name)
    print("Function Arguments:", function_args)

    ## Find the correspoding function and call it with the given arguments
    available_functions = {
        "rag_pipeline_func": rag_pipeline_func,
        "get_current_weather": get_current_weather,
    }
    function_to_call = available_functions[function_name]

    function_response = function_to_call(**function_args)
    print("Function Response:", function_response)
    return function_name, str(function_response.values())


def get_messages(system_prompt, query):
    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(query),
    ]
    return messages


def generate_function_response_chat(
    function_name, function_response, messages, chat_generator
):
    function_message = ChatMessage.from_function(
        content=json.dumps(function_response), name=function_name
    )
    messages.append(function_message)

    response = chat_generator.run(messages=messages)
    return response


def cli():
    global RAG_PIPE
    # example_create_chat_generator()
    document_store = create_indexed_documents()
    RAG_PIPE = create_rag_pipeline(document_store)

    tools = [get_rag_pipeline_func_spec(), get_current_weather_spec()]

    system_prompt = (
        "Don't make assumptions about what values to plug into "
        + "functions. Ask for clarification if a user request is ambiguous."
    )

    chat_generator = OpenAIChatGenerator(
        model="gpt-3.5-turbo", streaming_callback=print_streaming_chunk
    )

    query = "Can you tell me where Mark lives?"
    messages = get_messages(system_prompt, query)
    response = chat_generator.run(
        messages=messages, generation_kwargs={"tools": tools[0]}
    )
    print(f"{query=}")
    fn_name, fn_response = get_function_response(response)
    response = generate_function_response_chat(
        fn_name, fn_response, messages, chat_generator
    )
    print(response)

    query = "What is the weather like in Madrid?"
    messages = get_messages(system_prompt, query)
    response = chat_generator.run(
        messages=messages, generation_kwargs={"tools": tools[1]}
    )
    print(f"{query=}")
    fn_name, fn_response = get_function_response(response)
    response = generate_function_response_chat(
        fn_name, fn_response, messages, chat_generator
    )
    print(response)


def chatbot_with_fc(message, history):
    _ = history
    return "answer to " + message


def web():
    demo = gr.ChatInterface(
        fn=chatbot_with_fc,
        examples=[
            "Can you tell me where Giorgio lives?",
            "What's the weather like in Madrid?",
            "Who lives in London?",
            "What's the weather like where Mark lives?",
        ],
        title="Ask me about weather or where people live!",
    )
    demo.launch()


def main():
    fire.Fire({"cli": cli, "web": web})


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
