SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: openai-example
openai-example:  ## run openai-example
	poetry run python openai-example.py

.PHONY: jupyter
jupyter: ## start jupyter lab
	poetry run jupyter lab --ip=127.0.0.1 --no-browser --notebook-dir=$(SCRIPT_DIR)
