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
	docker compose up -d
	poetry install

.PHONY: env-rm
env-rm:
	docker compose rm -s -v -f
	poetry env remove $$(poetry env info -e)

PHONY: starter-openai
starter-openai: env-up  ## llamaindex openai starter
	poetry run python starter-openai.py

PHONY: default-vector-store
default-vector-store: env-up  ## query default vector store
	poetry run python ./vector-store-options.py $@

PHONY: redis-vector-store
redis-vector-store: env-up  ## query redis vector store
	poetry run python ./vector-store-options.py $@

PHONY: redis-custom-schema
redis-custom-schema: env-up  ## query redis custom schema
	poetry run python ./vector-store-options.py $@

PHONY: vector-svm-regression
vector-svm-regression: env-up  ## simple vector search with regression
	poetry run python simple-vector-store.py $@

PHONY: vector-async-index
vector-async-index: env-up  ## simple vector index using async
	poetry run python simple-vector-store.py $@

PHONY: vector-maximum-marginal-rel
vector-maximum-marginal-rel: env-up  ## simple vector relevance threshold
	poetry run python simple-vector-store.py $@

PHONY: faiss-vector-store
faiss-vector-store: env-up  ## simple vector relevance threshold
	poetry run python $@.py

PHONY: docarray-inmemory-vector
docarray-inmemory-vector:  ## docarry inmemory vector store
	poetry run python docarray-vector-store.py $@

PHONY: docarray-hnsw-vector
docarray-hnsw-vector:  ## docarry hnsw vector store
	poetry run python docarray-vector-store.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name 'temp_storage' -type d -exec rm -rf '{}' +
