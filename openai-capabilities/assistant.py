"""
GPT using assistants
https://platform.openai.com/docs/assistants/overview?context=without-streaming
https://cookbook.openai.com/examples/assistants_api_overview_python
"""

import json
import logging
import pathlib
import tempfile
import time
from urllib import request

import fire
import openai
from openai._models import BaseModel
from openai.types.beta.assistant import Assistant

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

GPT_MODEL = "gpt-4-turbo-preview"


def create_assistant() -> Assistant:
    assistant = openai.OpenAI().beta.assistants.create(
        name="Math Tutor",
        instructions=(
            "You are a personal math tutor."
            " Answer questions briefly in a sentence or less."
        ),
        model=GPT_MODEL,
    )
    return assistant


def create_assistant_file(file_ids) -> Assistant:
    assistant = openai.OpenAI().beta.assistants.create(
        name="Math Tutor",
        instructions=(
            "You are a personal math tutor."
            " Answer questions briefly in a sentence or less."
        ),
        model=GPT_MODEL,
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
        file_ids=file_ids,
    )
    return assistant


def create_thread():
    thread = openai.OpenAI().beta.threads.create()
    return thread


def create_message(thread, content):
    message = openai.OpenAI().beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content,
    )
    return message


def create_run(thread, assistant):
    run = openai.OpenAI().beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    return run


def print_objects(*args):
    marker = "=" * 3
    for arg in args:
        assert isinstance(arg, BaseModel), f"{arg} is not an object"
        print(f"{marker} {arg.__class__.__name__}", end="")
        print(json.loads(arg.model_dump_json()))


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = openai.OpenAI().beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(1)
    return run


def list_messages(thread):
    messages = openai.OpenAI().beta.threads.messages.list(thread_id=thread.id)
    return messages


def submit_message(assistant, thread, content):
    create_message(thread, content)
    return create_run(thread, assistant)


def create_thread_and_run(content, assistant):
    thread = create_thread()
    run = submit_message(assistant, thread, content)
    return thread, run


def print_messages(messages):
    for m in messages:
        print(f"=== {m.role}: {m.content[0].text.value}")
    print()


def assistant_no_tools():
    "use assistant with no tools"
    assistant = create_assistant()

    thread1, run1 = create_thread_and_run(
        "I need to solve the equation `3x + 11 = 14`. Can you help me?",
        assistant,
    )
    thread2, run2 = create_thread_and_run(
        "Could you explain linear algebra to me?", assistant
    )
    thread3, run3 = create_thread_and_run(
        "I don't like math. What can I do?", assistant
    )

    run1 = wait_on_run(run1, thread1)
    print_messages(list_messages(thread1))

    run2 = wait_on_run(run2, thread2)
    print_messages(list_messages(thread2))

    run3 = wait_on_run(run3, thread3)
    print_messages(list_messages(thread3))


def assistant_code_interpreter():
    "use assistant with code interpreter"
    assistant = create_assistant()
    assistant = openai.OpenAI().beta.assistants.update(
        assistant.id,
        tools=[{"type": "code_interpreter"}],
    )
    thread, run = create_thread_and_run(
        "Generate the first 20 fibbonaci numbers with code.", assistant
    )

    run = wait_on_run(run, thread)
    print_messages(list_messages(thread))

    run_steps = openai.OpenAI().beta.threads.runs.steps.list(
        thread_id=thread.id, run_id=run.id, order="asc"
    )
    for step in run_steps.data:
        step_details = step.step_details
        print(step_details.model_dump_json(indent=4))


def download_pdf(url):
    """download pdf from url and return bytes

    Download pdf file

    Args:
        url (str): url of file to download
    """
    pdf_file = request.urlopen(url)
    pdf_data = pdf_file.read()
    pdf_file.close()
    return pdf_data


def assistant_retrieval():
    "use assistant with retrieval to analyze a document"

    doc_name = "language_models_are_unsupervised_multitask_learners.pdf"
    base_url = "https://d4mucfpksywv.cloudfront.net/better-language-models/"
    pdf_data = download_pdf(base_url + doc_name)

    log.info("Size of data {}".format(len(pdf_data)))

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file inside the temporary directory
        temp_file = pathlib.Path(temp_dir) / doc_name
        temp_file.write_bytes(pdf_data)

        file = openai.OpenAI().files.create(
            file=temp_file,
            purpose="assistants",
        )
        print(file.model_dump_json(indent=4))

        file_ids = [file.id]
        assistant = create_assistant_file(file_ids)
        print(assistant.model_dump_json(indent=4))

        thread, run = create_thread_and_run(
            "What are some cool math concepts behind this ML paper pdf?"
            " Explain in two sentences.",
            assistant,
        )
        run = wait_on_run(run, thread)
        print_messages(list_messages(thread))


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "assistant-no-tools": assistant_no_tools,
            "assistant-code-interpreter": assistant_code_interpreter,
            "assistant-retrieval": assistant_retrieval,
        }
    )


if __name__ == "__main__":
    main()
