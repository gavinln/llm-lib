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
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: hello
hello: env-up  ## hello
	echo hello

.PHONY: brute-force-search
brute-force-search: env-up  ## brute force search
	poetry run python getting-started.py $@

.PHONY: voronoi-search
voronoi-search: env-up  ## voronoi-search
	poetry run python getting-started.py $@

.PHONY: low-memory-search
low-memory-search: env-up  ## low-memory-search
	poetry run python getting-started.py $@

.PHONY: clustering
clustering: env-up  ## clustering
	poetry run python faiss-building-blocks.py $@

.PHONY: pca
pca: env-up  ## principal component analysis
	poetry run python faiss-building-blocks.py $@

.PHONY: quantization
quantization: env-up  ## quantization
	poetry run python faiss-building-blocks.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
