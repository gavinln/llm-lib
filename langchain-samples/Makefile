SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

# highlight the Makefile targets
# @grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

.PHONY: tmux
tmux:  ## run tmux
	tmuxp load .tmuxp.yaml

.PHONY: fix
fix:  ## run isort and ruff on python code
	isort --float-to-top *.py
	ruff check --fix *.py
	poetry run black -l 79 *.py

.PHONY: env-up
env-up:
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: quickstart-openai
quickstart-openai: env-up  ## simple example with openai
	poetry run python $@.py

.PHONY: retrieval-chain
retrieval-chain: env-up  ## retrieve document for context
	poetry run python $@.py

.PHONY: rag-quickstart
rag-quickstart: env-up  ## use RAG to answer question
	poetry run python rag.py $@

.PHONY: rag-chat-history
rag-chat-history: env-up  ## RAG with chat history
	poetry run python rag.py $@

.PHONY: rag-streaming
rag-streaming: env-up  ## RAG with streaming
	poetry run python rag.py $@

.PHONY: rag-citations
rag-citations: env-up  ## get citations from Wikipedia
	poetry run python rag.py $@

.PHONY: structured-one-entity
structured-one-entity: env-up  ## extract structured data; one entity
	poetry run python ./structured-output.py $@

.PHONY: structured-many-entities
structured-many-entities: env-up  ## extract structured data; many entities
	poetry run python ./structured-output.py $@

.PHONY: tool-use
tool-use: env-up  ## LLM tool use
	poetry run python ./tool_agent_use.py $@

.PHONY: agent-used
agent-use: env-up  ## LLM agent use
	poetry run python ./tool_agent_use.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name '.ruff_cache' -type d -exec rm -rf '{}' +
