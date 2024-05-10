SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

# highlight the Makefile targets
# @grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

.PHONY: docker-daemon-start
docker-daemon-start:
	echo sudo service docker start

.PHONY: docker-compose-test
docker-compose-test: docker-daemon-start  ## example of using docker compose
	echo open browser to http://127.0.0.1:8000.
	docker compose --project-directory ./composetest/ up

.PHONY: docker-compose-test-rm
docker-compose-test-rm:  ## remove example docker compose
	docker compose --project-directory ./composetest/ rm -s -v

.PHONY: data-structure-store
data-structure-store:  ## redis data structure store
	docker compose --project-directory ./redis-examples/ up -d
	poetry run python data-structure-store.py

.PHONY: document-database
document-database:  ## redis document database
	docker compose --project-directory ./redis-examples/ up -d
	poetry run python document-database.py

.PHONY: vector-database
vector-database:  ## redis vector database
	docker compose --project-directory ./redis-examples/ up -d
	poetry run python vector-database.py

.PHONY: sentence-xfmer
sentence-xfmer:  ## sentence transformer examples
	poetry run python sentence-transformer-example.py

.PHONY: clean
clean: ## remove temporary files
	echo clean
