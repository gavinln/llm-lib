SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: env-up
env-up:
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: chat-basic
chat-basic: env-up  ## run chat-basic flask example
	pushd ./examples/chat-basic && poetry run flask run && popd

.PHONY: assistant-basic
assistant-basic:  ## run assistant-basic command line example
	poetry run python3 ./examples/assistant-basic/assistant.py

.PHONY: assistant-functions
assistant-functions: env-up  ## run assistant-functions command line example using tools
	echo Try: "What is the current weather?"
	poetry run python3 ./examples/assistant-functions/functions.py

.PHONY: assistant-flask
assistant-flask: env-up  ## run assistant-flask web example
	echo Upload the file people.csv and ask questions about the people in the file
	pushd ./examples/assistant-flask && poetry run flask run && popd

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -rf output/
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
