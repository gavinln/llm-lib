SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: jupyter
jupyter:  ## start jupyter lab
	poetry run jupyter lab --port=8888 --ip=127.0.0.1 --no-browser --notebook-dir=$(SCRIPT_DIR)

.PHONY: open-ai-example
open-ai-example:  ## run minimum open-ai-example
	poetry run python openai-examples/open-ai-example.py

.PHONY: aider
aider:  ## run aider to modify code using AI
	aider -4 --no-auto-commits --no-dirty-commits

.PHONY: clean
clean:  ## remove temporary directories in python envs
	rm -rf .aider.*
	@make -C ./deeplake-samples/ clean
	@make -C ./dspy-samples/ clean
	@make -C ./faiss-samples/ clean
	@make -C ./gorilla-samples/ clean
	@make -C ./guidance-samples/ clean
	@make -C ./haystack-samples/ clean
	@make -C ./langchain-samples/ clean
	@make -C ./llamaindex-samples/ clean
	@make -C ./openai-capabilities/ clean
	@make -C ./openai-examples/ clean
	@make -C ./openai-prompts/ clean
	@make -C ./openai-quickstart-python/ clean
	@make -C ./openai-website-qa/ clean
	@make -C ./phidata-samples/ clean
	@make -C ./redis-examples/ clean
