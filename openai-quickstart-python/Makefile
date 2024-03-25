SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: chat-basic
chat-basic:  ## run chat-basic flask example
	pushd ./openai-quickstart-python/examples/chat-basic && poetry run flask run && popd

.PHONY: assistant-basic
assistant-basic:  ## run assistant-basic command line example
	poetry run python ./openai-quickstart-python/examples/assistant-basic/assistant.py

.PHONY: assistant-functions
assistant-functions:  ## run assistant-functions command line example using tools
	echo Try: "What is the current weather?"
	poetry run python ./openai-quickstart-python/examples/assistant-functions/functions.py

.PHONY: assistant-flask
assistant-flask:  ## run assistant-flask web example
	echo Upload the file people.csv and ask questions about the people in the file
	pushd ./openai-quickstart-python/examples/assistant-flask && poetry run flask run && popd
