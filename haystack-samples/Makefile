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

.PHONY: starter-website-chat
starter-website-chat: env-up  ## starter examples
	poetry run python haystack-starter.py $@

.PHONY: starter-fetch-query
starter-fetch-query: env-up  ## starter examples
	poetry run python haystack-starter.py $@

.PHONY: rag-predefined
rag-predefined: env-up  ## starter examples
	poetry run python haystack-rag.py $@

.PHONY: rag-pipeline
rag-pipeline: env-up  ## starter examples
	poetry run python haystack-rag.py $@

.PHONY: chroma-info
chroma-info: env-up  ## chromadb info
	poetry run python chroma-starter.py $@

.PHONY: chroma-collections
chroma-collections: env-up  ## chromadb collections
	poetry run python chroma-starter.py $@

.PHONY: chroma-query
chroma-query: env-up  ## chromadb query
	poetry run python chroma-starter.py $@

.PHONY: rag-qa-pipeline
rag-qa-pipeline: env-up  ## answer questions using RAG
	poetry run python $@.py

.PHONY: metadata-filtering
metadata-filtering: env-up  ## use metadata filtering
	poetry run python $@.py

.PHONY: preprocess-files
preprocess-files: env-up  ## load different file types in doc store
	poetry run python $@.py

.PHONY: embedding-metadata
embedding-metadata: env-up  ## embedding in metadata
	poetry run python $@.py

.PHONY: serialize-pipelines
serialize-pipelines: env-up  ## serialize pipelines to yaml
	poetry run python $@.py

.PHONY: extractive-pipeline
extractive-pipeline: env-up  ## extract verbatim text
	poetry run python $@.py

.PHONY: hybrid-retrieval-pipeline
hybrid-retrieval-pipeline: env-up  ## embedding & keyword retrieval
	poetry run python $@.py

.PHONY: structured-output
structured-output: env-up  ## generate llm output in JSON format
	poetry run python $@.py

.PHONY: classify-documents
classify-documents: env-up  ## classify documents by human language
	poetry run python $@.py

.PHONY: evaluate-pipelines
evaluate-pipelines: env-up  ## evaluate rag pipeline
	poetry run python $@.py

.PHONY: chat-app-tools-cli
chat-app-tools-cli: env-up  ## chat app with tools - cli
	poetry run python chat-app-tools.py cli

.PHONY: chat-app-tools-web
chat-app-tools-web: env-up  ## chat app with tools - web
	poetry run python chat-app-tools.py web

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name 'temp_storage' -type d -exec rm -rf '{}' +
