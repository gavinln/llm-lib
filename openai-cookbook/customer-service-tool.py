"""
https://cookbook.openai.com/examples/using_tool_required_for_customer_service
"""

import logging
import pathlib
import sys
import tempfile

import fire
from joblib import Memory

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

memory = Memory(tempfile.gettempdir(), verbose=0)

# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-4-turbo"


def customer_service_tool():
    print("NOT IMPLEMENTED")


def main():
    print(sys.version)
    fire.Fire({"customer-service-tool": customer_service_tool})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
