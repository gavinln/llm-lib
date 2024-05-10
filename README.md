# Libraries for large language models: llm-lib

## ChatGPT interactive use

Login at the following address

https://chat.openai.com/auth/login

### Introduction

https://platform.openai.com/docs/introduction/overview

* Content generation
* Summarization
* Classification, categorization, and sentiment analysis
* Data extraction
* Translation

## Run openai-quickstart-python

See [openai-quickstart-python](./openai-quickstart-python/README.md)

## Run openai-examples

1. Setup API key

```
source ./do_not_checkin/openai_key.sh
```

2. Run openai-example in a poetry virtual environment

```
make open-ai-example
```

## Run openai-cookbook

1. Clone the project

```
git clone https://github.com/openai/openai-cookbook
```

2. Change to the project root

### Datasets

#### Fine food reviews

There are two files

```
./openai-cookbook/examples/data/fine_food_reviews_1k.csv
./openai-cookbook/examples/data/fine_food_reviews_with_embeddings_1k.csv
```

The following notebooks use the above two data files.

* Obtain_dataset.ipynb - read the first file and create the second file with embeddings
* Visualizing_embeddings_in_2D.ipynb - visualize embeddings using tSNE
* Visualizing_embeddings_with_Atlas.ipynb - requires Nomic login https://atlas.nomic.ai/
* Clustering.ipynb - kMeans clustering and tSNE
* Visualizing_embeddings_in_W&B.ipynb - requires Weights and Biases
* Zero-shot_classification_with_embeddings.ipynb - uses cosine distance of embeddings to classify positive/negative
* Classification_using_embeddings.ipynb - uses RandomForest classifier with embeddings to learn ratings
* Regression_using_embeddings.ipynb - uses RandomForest regressor to learn ratings
* Semantic_text_search_using_embeddings.ipynb

## Ollama

[Ollama][100] is used to run large language models locally.

[100]: https://github.com/ollama/ollama

Install Ollama using the command

```
curl -fsSL https://ollama.com/install.sh | sh
```

## Llama index

https://github.com/run-llama/llama_index

## Semantic kernel

https://github.com/microsoft/semantic-kernel

## Jupyterlab in a different virtual environment

https://samedwardes.com/2022/10/23/best-jupyter-lab-install/

1. Install jupyterlab using pipx

```
pipx install jupyterlab --include-deps
```

2. Check pipx binary directories are added to PATH

```
pipx ensurepath
```

3. Check for the jupyterlab environment

```
pipx list --short
```

4. List available kernels

```
jupyter kernelspec list
```

5. Switch to the virtual environment

```
poetry shell
```

6. Install ipykernel in the environment

```
poetry add ipykernel
```

7. Register the kernel

```
python -m ipykernel install --user --display-name ${PWD} --name ${PWD##*/}
```

8. List available kernels. A new one is added

```
jupyter kernelspec list
```

## Links

[Prompt engineering guide][1000]

[1000]: https://github.com/dair-ai/Prompt-Engineering-Guide

Chat [GPT login][1010]

[1010]: https://chat.openai.com/auth/login

https://github.com/langchain-ai/langchain

Building applications with Large Language Models (LLMs)


### Getting started with LLMs using LangChain

https://www.pinecone.io/learn/langchain-intro/

### Videos

https://www.youtube.com/@DataIndependent/videos

https://www.youtube.com/@echohive/videos

https://www.youtube.com/watch?v=aywZrzNaKjs

### Other

[How to build a semantic search][1100]

[1100]: https://haystack.deepset.ai/blog/how-to-build-a-semantic-search-engine-in-python

[Question answering chatbot][1110]

[1110]: https://github.com/jerpint/buster

[Byte pair encoding][1120]

[1120]: https://huggingface.co/course/chapter6/5

#### Github libraries

https://github.com/hwchase17/langchain - 61k stars

https://github.com/nomic-ai/gpt4all - 52k stars

https://github.com/microsoft/guidance - 13k stars

https://github.com/stanfordnlp/dspy - 2.4k stars

https://github.com/jerryjliu/llama_index - 21k stars

#### Comparison of libraries

https://medium.com/badal-io/exploring-langchain-and-llamaindex-to-achieve-standardization-and-interoperability-in-large-2b5f3fabc360
