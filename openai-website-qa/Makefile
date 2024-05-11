SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: env-up
env-up:
	poetry install

.PHONY: env-rm
env-rm:
	poetry env remove $$(poetry env info -e)

.PHONY: web-crawler
web-crawler: env-up  ## scrape text from a website
	poetry run python web_crawler.py

.PHONY: process-text
process-text: env-up  ## process scraped website text
	poetry run python process_text.py

.PHONY: answer-question
answer-question: env-up  ## answer by finding closest embedding
	poetry run python answer_question.py

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -rf text/
	rm -f poetry.lock
