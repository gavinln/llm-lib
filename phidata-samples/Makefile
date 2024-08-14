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

.PHONY: web-search
web-search:  ## duckduckgo web search
	poetry run python $@.py

.PHONY: finance-assistant
finance-assistant:  ## yahoo finance
	poetry run python $@.py

.PHONY: research
research:  ## web search to write article
	poetry run python $@.py

.PHONY: start-pgvector
start-pgvector:  ## start pgvector with docker
	docker compose up -d
	# docker run -d \
	#   -e POSTGRES_DB=ai \
	#   -e POSTGRES_USER=ai \
	#   -e POSTGRES_PASSWORD=ai \
	#   -e PGDATA=/var/lib/postgresql/data/pgdata \
	#   -v pgvolume:/var/lib/postgresql/data \
	#   -p 5532:5432 \
	#   --name pgvector \
	#   phidata/pgvector:16

.PHONY: stop-pgvector
stop-pgvector:  ## stop pgvector with docker
	docker compose rm -s -f
	# docker stop $$(docker ps -q -f "ancestor=phidata/pgvector:16")

.PHONY: rag-assistant
rag-assistant:  ## Use RAG with pdf to answer question
	poetry run python $@.py

.PHONY: auto-assistant
auto-assistant:  ## Let LLM use RAG to answer question
	poetry run python $@.py

.PHONY: pdf-assistant
pdf-assistant:  ## assistant with memory, knowledge, tools
	poetry run python $@.py

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
	find . -name 'temp_storage' -type d -exec rm -rf '{}' +
