SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

# highlight the Makefile targets
# @grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

PROJECT := openai-cookbook
VENV := ~/.cache/venv/$(PROJECT)

.PHONY: fix
fix:  ## run isort and ruff on python code
	isort --float-to-top *.py
	ruff check --fix *.py
	black -l 79 *.py

.PHONY: env-up
env-up:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync

.PHONY: env-rm
env-rm:
	rm -rf $(VENV)

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '.ruff_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +

.PHONY: nvim
nvim:  ## nvim
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run nvim -S Session.vim

.PHONY: guardrail-input-good
guardrails-input-good: env-up  ## guardrails with good input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py input-good

.PHONY: guardrail-input-bad
guardrails-input-bad: env-up  ## guardrails with bad input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py input-bad

.PHONY: output-guardrails
output-guardrails: env-up  ## output and input guardrails
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py output-guardrails

.PHONY: logprobs-classification
logprobs-classification: env-up  ## classification with log probabilities
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python logprobs.py classification

.PHONY: reduce-hallucinations
reduce-hallucinations: env-up  ## reduce hallucations with log probabilities
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python logprobs.py reduce-hallucinations

.PHONY: autocomplete
autocomplete: env-up  ## autocomplete with log probabilities
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python logprobs.py autocomplete

.PHONY: perplexity
perplexity: env-up  ## perplexity with log probabilities
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python logprobs.py perplexity

.PHONY: moderation-input-good
moderation-input-good: env-up  ## moderation api - good input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python moderation.py $@

.PHONY: moderation-input-bad
moderation-input-bad: env-up  ## moderation api - bad input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python moderation.py $@

.PHONY: moderation-custom-good
moderation-custom-good: env-up  ## custom moderation - good input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python moderation.py $@

.PHONY: moderation-custom-bad
moderation-custom-bad: env-up  ## custom moderation api - bad input
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python moderation.py $@

.PHONY: csv-with-prompt
csv-with-prompt: env-up  ## create csv with a structured prompt
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python synthetic-data.py $@

.PHONY: csv-with-python
csv-with-python: env-up  ## create csv with a Python program
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python synthetic-data.py $@

.PHONY: csv-multitable-with-python
csv-multitable-with-python: env-up  ## create multiple csv tables
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python synthetic-data.py $@

.PHONY: text-with-prompt
text-with-prompt: env-up  ## create plain text with a prompt
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python synthetic-data.py $@

.PHONY: text-imbalanced
text-imbalanced: env-up  ## handle imbalanced text
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python synthetic-data.py $@

.PHONY: doc-summary
doc-summary: env-up  ## summarize document
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python summarize-long-docs.py $@

.PHONY: doc-summary-points
doc-summary-points: env-up  ## summarize document using points
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python summarize-long-docs.py $@

.PHONY: doc-summary-recursive
doc-summary-recursive: env-up  ## summarize document recursively
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python summarize-long-docs.py $@

.PHONY: customer-service-tool
customer-service-tool: env-up  ## customer service using tools
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python customer-service-tool.py $@
