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

.PHONY: env-up
env-up:
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: pytest
pytest: env-up  ## run Python tests
	poetry run pytest ./process_openai_examples.py
	poetry run pytest ./embedding-long-texts.py

.PHONY: print-tags
print-tags: env-up  ## print tags from openai examples
	poetry run python ./process_openai_examples.py $@

.PHONY: print-examples-stats
print-examples-stats: env-up  ## print openai examples stats
	poetry run python ./process_openai_examples.py $@

.PHONY: vector-in-titles
vector-in-titles: env-up  ## print openai examples with vector
	poetry run python ./process_openai_examples.py $@

.PHONY: redis-in-titles
redis-in-titles: env-up  ## print openai examples with redis
	poetry run python ./process_openai_examples.py $@

.PHONY: unit-test-writing
unit-test-writing: env-up  ## using openai to create unit tests
	poetry run python $@.py

.PHONY: embedding-long-texts
embedding-long-texts: env-up  ## use chunks for long texts
	poetry run python $@.py

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -f poetry.lock
