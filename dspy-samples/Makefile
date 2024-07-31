SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

# highlight the Makefile targets
# @grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

.PHONY: env-up
env-up:
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: hello
hello: env-up  ## hello
	echo hello

.PHONY: direct-llm-call
direct-llm-call: env-up  ## directly call llm
	poetry run python language-models.py $@

.PHONY: dspy-llm-call
dspy-llm-call: env-up  ## call llm with dspy signature
	poetry run python language-models.py $@

.PHONY: multiple-llms
multiple-llms: env-up  ## call multiple llms
	poetry run python language-models.py $@

.PHONY: multiple-outputs
multiple-outputs: env-up  ## generate multiple outputs
	poetry run python language-models.py $@

.PHONY: remote-llms
remote-llms: env-up  ## call remote llms
	poetry run python language-models.py $@

.PHONY: local-llms
local-llms: env-up  ## call local llm
	poetry run python language-models.py $@

.PHONY: inline-signatures
inline-signatures: env-up  ## signature defined with a string
	poetry run python signatures.py $@

.PHONY: class-signatures
class-signatures: env-up  ## signature defined with a class
	poetry run python signatures.py $@

.PHONY: jupyter
jupyter:  ## start jupyter lab
	poetry run jupyter lab --port=8888 --ip=127.0.0.1 --no-browser --notebook-dir=$(SCRIPT_DIR)/notebooks

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
