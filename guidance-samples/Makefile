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

.PHONY: env-up
env-up:
	# docker compose up -d
	poetry install

.PHONY: env-rm
env-rm:
	# docker compose rm -s -v -f
	poetry env remove $$(poetry env info -e)

.PHONY: starter-guidance
starter-guidance: env-up  ## starter examples
	poetry run python $@.py

.PHONY: basics-simple-gen
basics-simple-gen: env-up  ## simple generation
	poetry run python basics-gen.py $@

.PHONY: basics-templates
basics-templates: env-up  ## templates
	poetry run python basics-gen.py $@

.PHONY: basics-capturing
basics-capturing: env-up  ## capture variables
	poetry run python basics-gen.py $@

.PHONY: basics-functions
basics-functions: env-up  ## function encapsulation
	poetry run python basics-gen.py $@

.PHONY: constrained-select
constrained-select: env-up  ## select constrained generation
	poetry run python constrained_gen.py $@

.PHONY: constrained-interleave
constrained-interleave: env-up  ## interleave constrained generation
	poetry run python constrained_gen.py $@

.PHONY: constrained-regex
constrained-regex: env-up  ## regex constrained generation
	poetry run python constrained_gen.py $@

.PHONY: constrained-cfg
constrained-cfg: env-up  ## context free grammar constrained gen
	poetry run python constrained_gen.py $@

.PHONY: constrained-cfg2
constrained-cfg2: env-up  ## context free grammar constrained gen
	poetry run python constrained_gen.py $@

.PHONY: constrained-cfg3
constrained-cfg3: env-up  ## context free grammar constrained gen
	poetry run python constrained_gen.py $@

.PHONY: stateful-basics
stateful-basics: env-up  ## stateful generation
	poetry run python stateful_gen.py $@

.PHONY: stateful-re-act
stateful-re-act: env-up  ## stateful ReAct - Reasoning Action
	poetry run python stateful_gen.py $@

.PHONY: stateful-tools1
stateful-tools1: env-up  ## stateful Reasoning Action Tools
	poetry run python stateful_gen.py $@

.PHONY: stateful-tools2
stateful-tools2: env-up  ## stateful Reasoning Action Tools
	poetry run python stateful_gen.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name 'temp_storage' -type d -exec rm -rf '{}' +
