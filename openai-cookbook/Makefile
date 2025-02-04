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
	find . -name '__pycache__' -type d -exec rm -rf '{}' +

.PHONY: nvim
nvim:  ## nvim
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run nvim -S Session.vim

.PHONY: guardrail-input-good
guardrails-input-good: env-up  ## run guardrails
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py input-good

.PHONY: guardrail-input-bad
guardrails-input-bad: env-up  ## run guardrails
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py input-bad

.PHONY: output-guardrails
output-guardrails: env-up  ## run guardrails
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python guardrails.py output-guardrails
