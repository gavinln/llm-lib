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

.PHONY: docker-daemon-start
docker-daemon-start:
	echo sudo service docker start

.PHONY: docker-compose-test-up
docker-compose-test-up: docker-daemon-start  ## example of using docker compose
	echo open browser to http://127.0.0.1:8000.
	poetry run docker compose --project-directory ./composetest/ up

.PHONY: docker-compose-test-rm
docker-compose-test-rm:  ## remove example docker compose
	poetry run docker compose --project-directory ./composetest/ rm -s -v

.PHONY: env-up
env-up:
	docker compose up -d
	poetry install

.PHONY: env-rm
env-rm:
	docker compose rm -s -v -f
	poetry env remove $$(poetry env info -e)

.PHONY: data-structure-store
data-structure-store: env-up  ## redis data structure store
	poetry run python data-structure-store.py

.PHONY: document-database
document-database: env-up  ## redis document database
	poetry run docker compose up -d
	poetry run python document-database.py

.PHONY: vector-database
vector-database: env-up  ## redis vector database
	poetry run docker compose up -d
	poetry run python vector-database.py

.PHONY: sentence-xfmer
sentence-xfmer:  ## sentence transformer examples
	poetry run python sentence-transformer-example.py

.PHONY: vector-search-existing
vector-search-existing:  ## vector search with existing embedding
	echo "Does NOT WORK correctly"
	poetry run docker compose up -d
	poetry run python $@.py

.PHONY: vector-search-context
vector-search-context:  ## vector search for context
	poetry run docker compose up -d
	poetry run python $@.py

.PHONY: vector-search-hybrid
vector-search-hybrid:  ## vector hybrid search
	poetry run docker compose up -d
	poetry run python $@.py

.PHONY: vector-search
vector-search:  ## vector search
	poetry run docker compose up -d
	poetry run python $@.py

.PHONY: redisvl-basic
redisvl-basic:  ## basic redis vector lib
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-vector-query
redisvl-vector-query:  ## vector query
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-tag-filters
redisvl-tag-filters:  ## vector query with tag filters
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-numeric-filters
redisvl-numeric-filters:  ## vector query with numeric filters
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-text-filters
redisvl-text-filters:  ## vector query with text filter
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-combined-filters
redisvl-combined-filters:  ## vector query: text, numeric filters
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-filter-queries
redisvl-filter-queries:  ## filter query, no vector query
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-count-queries
redisvl-count-queries:  ## count query, no vector query
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: redisvl-range-queries
redisvl-range-queries:  ## vector query with range
	poetry run docker compose up -d
	poetry run python redis-vector-lib-samples.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
