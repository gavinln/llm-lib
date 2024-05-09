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

.PHONY: pytest
pytest:  ## run Python tests
	poetry run pytest ./process_openai_examples.py
	poetry run pytest ./embedding-long-texts.py

.PHONY: print-tags
print-tags:  ## print tags from openai examples
	poetry run python ./process_openai_examples.py $@

.PHONY: print-examples-stats
print-examples-stats:  ## print openai examples stats
	poetry run python ./process_openai_examples.py $@

.PHONY: vector-in-titles
vector-in-titles:  ## print openai examples with vector
	poetry run python ./process_openai_examples.py $@

.PHONY: redis-in-titles
redis-in-titles:  ## print openai examples with redis
	poetry run python ./process_openai_examples.py $@

.PHONY: unit-test-writing
unit-test-writing:  ## using openai to create unit tests
	poetry run python $@.py

.PHONY: embedding-long-texts
embedding-long-texts:  ## use chunks for long texts
	poetry run python $@.py

# .PHONY: docker-compose-test
# docker-compose-test: docker-daemon-start  ## example of using docker compose
# 	docker compose --project-directory ./redis-examples/composetest/ up
# 	echo open browser to http://127.0.0.1:8000.
#
# .PHONY: docker-compose-test-rm
# docker-compose-test-rm:  ## remove example docker compose
# 	docker compose --project-directory ./redis-examples/composetest/ rm -s -v
#
# .PHONY: data-structure-store
# data-structure-store:  ## redis data structure store
# 	docker compose --project-directory ./redis-examples/ up -d
# 	poetry run python ./redis-examples/data-structure-store.py
#
# .PHONY: document-database
# document-database:  ## redis document database
# 	docker compose --project-directory ./redis-examples/ up -d
# 	poetry run python ./redis-examples/document-database.py
#
# .PHONY: vector-database
# vector-database:  ## redis vector database
# 	docker compose --project-directory ./redis-examples/ up -d
# 	poetry run python ./redis-examples/vector-database.py
