SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SHELL = /bin/bash

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: save-food-reviews-embeddings
save-food-reviews-embeddings:  ## get and save food reviews embeddings
	poetry run python embeddings.py save-food-reviews-embeddings

.PHONY: reduce-embeddings-dim
reduce-embeddings-dim:  ## reduce embeddings by normalizing 
	poetry run python embeddings.py reduce-embeddings-dim

.PHONY: answer-question-using-context
answer-question-using-context:  ## provide context article to answer question
	poetry run python embeddings.py answer-question-using-context

.PHONY: search-reviews
search-reviews:  ## search reviews using embeddings
	poetry run python embeddings.py search-reviews

.PHONY: save-ag-news-embeddings
save-ag-news-embeddings:  ## get and save ag news embeddings 
	poetry run python embeddings.py save-ag-news-embeddings

.PHONY: recommendation-news
recommendation-news:  ## recommend similar news articles
	poetry run python embeddings.py recommendation-news

.PHONY: visualization-reviews
visualization-reviews:  ## visualize distribution of reviews
	poetry run python embeddings.py visualization-reviews

.PHONY: regression-reviews
regression-reviews:  ## regression of reviews vs score
	poetry run python embeddings.py regression-reviews

.PHONY: classification-reviews
classification-reviews:  ## classification of reviews
	poetry run python embeddings.py classification-reviews

.PHONY: zero-shot-classification
zero-shot-classification:  ## classification of reviews; positive/negative
	poetry run python embeddings.py zero-shot-classification

.PHONY: user-product-embeddings
user-product-embeddings:  ## similarity between user/product embeddings
	poetry run python embeddings.py user-product-embeddings

.PHONY: clustering-reviews
clustering-reviews:  ## cluster using embeddings
	poetry run python embeddings.py clustering-reviews

.PHONY: chat-completion-text
chat-completion-text:  ## complete text with text output
	poetry run python text_generation.py chat-completion-text

.PHONY: chat-completion-json
chat-completion-json:  ## complete text with json output
	poetry run python text_generation.py chat-completion-json

.PHONY: count-tokens-local
count-tokens-local:  ## count tokens using the tiktoken library
	poetry run python text_generation.py count-tokens-local

.PHONY: count-tokens-remote
count-tokens-remote:  ## count tokens using an openai call
	poetry run python text_generation.py count-tokens-remote
