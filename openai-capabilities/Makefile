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

.PHONY: save-food-reviews-embeddings
save-food-reviews-embeddings: env-up  ## get and save food reviews embeddings
	poetry run python embeddings.py save-food-reviews-embeddings

.PHONY: reduce-embeddings-dim
reduce-embeddings-dim: env-up  ## reduce embeddings by normalizing
	poetry run python embeddings.py reduce-embeddings-dim

.PHONY: answer-question-using-context
answer-question-using-context: env-up  ## provide context article to answer question
	poetry run python embeddings.py answer-question-using-context

.PHONY: search-reviews
search-reviews: env-up  ## search reviews using embeddings
	poetry run python embeddings.py search-reviews

.PHONY: save-ag-news-embeddings
save-ag-news-embeddings: env-up  ## get and save ag news embeddings
	poetry run python embeddings.py save-ag-news-embeddings

.PHONY: recommendation-news
recommendation-news: env-up ## recommend similar news articles
	poetry run python embeddings.py recommendation-news

.PHONY: visualization-reviews
visualization-reviews: env-up ## visualize distribution of reviews
	poetry run python embeddings.py visualization-reviews

.PHONY: regression-reviews
regression-reviews: env-up ## regression of reviews vs score
	poetry run python embeddings.py regression-reviews

.PHONY: classification-reviews
classification-reviews: env-up ## classification of reviews
	poetry run python embeddings.py classification-reviews

.PHONY: zero-shot-classification
zero-shot-classification: env-up ## classification of reviews; positive/negative
	poetry run python embeddings.py zero-shot-classification

.PHONY: user-product-embeddings
user-product-embeddings: env-up ## similarity between user/product embeddings
	poetry run python embeddings.py user-product-embeddings

.PHONY: clustering-reviews
clustering-reviews: env-up  ## cluster using embeddings
	poetry run python embeddings.py clustering-reviews

.PHONY: chat-completion-text
chat-completion-text: env-up  ## complete text with text output
	poetry run python text_generation.py chat-completion-text

.PHONY: chat-completion-json
chat-completion-json: env-up  ## complete text with json output
	poetry run python text_generation.py chat-completion-json

.PHONY: count-tokens-local
count-tokens-local: env-up  ## count tokens using the tiktoken library
	poetry run python text_generation.py count-tokens-local

.PHONY: count-tokens-remote
count-tokens-remote: env-up  ## count tokens using an openai call
	poetry run python text_generation.py count-tokens-remote


.PHONY: basic-function-call
basic-function-call: env-up  ## basic function call
	poetry run python tool_call.py basic-function-call

.PHONY: ask-and-answer-place-question
ask-and-answer-place-question: env-up  ## ask and answer place question
	poetry run python tool_call.py ask-and-answer-place-question

.PHONY: ask-and-answer-day-question
ask-and-answer-day-question: env-up  ## ask and answer day question
	poetry run python tool_call.py ask-and-answer-day-question

.PHONY: ask-and-force-function-call
ask-and-force-function-call: env-up  ## ask and force function call
	poetry run python tool_call.py ask-and-force-function-call

.PHONY: ask-and-get-function-call
ask-and-get-function-call: env-up  ## ask and get function call
	poetry run python tool_call.py ask-and-get-function-call

.PHONY: ask-and-prevent-function-call
ask-and-prevent-function-call: env-up  ## ask and prevent function call
	poetry run python tool_call.py ask-and-prevent-function-call

.PHONY: parallel-function-call
parallel-function-call: env-up  ## parallel function call
	poetry run python tool_call.py parallel-function-call


.PHONY: sql-query-function-call
sql-query-function-call: env-up  ## answer user questions by generating sql queries
	poetry run python tool_call.py sql-query-function-call

.PHONY: assistant-no-tools
assistant-no-tools: env-up  ## run assistant with no tools
	poetry run python assistant.py $@

.PHONY: assistant-code-interpreter
assistant-code-interpreter: env-up  ## run assistant with code interpreter
	poetry run python assistant.py $@

.PHONY: assistant-retrieval
assistant-retrieval: env-up  ## run assistant with document retrieval
	poetry run python assistant.py $@

.PHONY: clean
clean: env-rm  ## remove temporary files
	rm -rf output/
	rm -f poetry.lock
	find . -name '.pytest_cache' -type d -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -exec rm -rf '{}' +
